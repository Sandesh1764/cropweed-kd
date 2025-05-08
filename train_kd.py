import argparse
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import DeepLab
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
import distiller

import torch
import torch.nn as nn


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # DataLoader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Teacher: DeepLabv3+ (ResNet101)
        self.t_net = DeepLab(num_classes=self.nclass,
                             backbone='resnet101',
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)
        teacher_path = getattr(args, "teacher_weights", None) or os.path.join(
            "pretrained", "deeplab-resnet101.pth.tar"
        )
        if os.path.isfile(teacher_path):
            ckpt = torch.load(teacher_path, map_location="cpu")
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.t_net.load_state_dict(state, strict=False)
            print(f"Loaded teacher weights from {teacher_path}")
        else:
            print(
                f"[warn] Teacher weights not found at {teacher_path}. "
                "Training will use a randomly initialized teacher — place a "
                "DeepLabV3+ ResNet-101 checkpoint there for real KD."
            )
        self.t_net.eval()
        for p in self.t_net.parameters():
            p.requires_grad = False

        # Student: DeepLab with lightweight backbone
        self.s_net = DeepLab(num_classes=self.nclass,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)

                # Distiller
        self.d_net = distiller.Distiller(self.t_net, self.s_net, args)

        # ─── Freeze ASPP global_avg_pool BatchNorm on both teacher & student BEFORE DataParallel ───
        for net in (self.t_net, self.s_net):
            gp = net.aspp.global_avg_pool
            for m in gp.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        self.d_net = distiller.Distiller(self.t_net, self.s_net, args)

        print('Teacher params:', sum(p.numel() for p in self.t_net.parameters()))
        print('Student params:', sum(p.numel() for p in self.s_net.parameters()))

        # Optimizer
        distill_params = [
            {'params': self.s_net.get_1x_lr_params(), 'lr': args.lr},
            {'params': self.s_net.get_10x_lr_params(), 'lr': args.lr * 10},
            {'params': self.d_net.Connectors.parameters(), 'lr': args.lr * 10},
        ]
        init_params = [{'params': self.d_net.Connectors.parameters(), 'lr': args.lr * 10}]
        self.optimizer = torch.optim.SGD(distill_params,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay,
                                         nesterov=args.nesterov)
        self.init_optimizer = torch.optim.SGD(init_params,
                                              momentum=args.momentum,
                                              weight_decay=args.weight_decay,
                                              nesterov=args.nesterov)

        # Loss criterion
        if args.use_balanced_weights:
            cw_path = os.path.join(Path.db_root_dir(args.dataset), f'{args.dataset}_classes_weights.npy')
            if os.path.isfile(cw_path):
                weight = np.load(cw_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Evaluator & LR scheduler
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # CUDA / DataParallel
        if args.cuda:
            self.s_net = nn.DataParallel(self.s_net).cuda()
            self.d_net = nn.DataParallel(self.d_net).cuda()

            # Freeze ASPP global_avg_pool BatchNorm
            teacher = self.d_net.module.t_net
            student = self.d_net.module.s_net
            for net in (teacher, student):
                gp = net.aspp.global_avg_pool
                for m in gp.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

        # Best prediction
        self.best_pred = 0.0
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.d_net.train()
        self.d_net.module.t_net.eval()
        self.d_net.module.s_net.train()

        tbar = tqdm(self.train_loader)
        optimizer = self.init_optimizer if epoch == 0 else self.optimizer

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(optimizer, i, epoch, self.best_pred)
            optimizer.zero_grad()

            output, pa_loss, pi_loss = self.d_net(image)
            loss_seg = self.criterion(output, target)
            if loss_seg.dim() != 0:
                loss_seg = loss_seg.mean()

            # Combine distillation and segmentation losses
            total_loss = loss_seg + pa_loss + pi_loss
            # ensure scalar
            total_loss = total_loss.mean()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            tbar.set_description(f'Train loss: {train_loss/(i+1):.3f}')

            train_loss += total_loss.item()
            tbar.set_description(f'Train loss: {train_loss/(i+1):.3f}')

        print(f'[Epoch {epoch}] Avg Loss: {train_loss/len(self.train_loader):.3f}')

    def validation(self, epoch):
        self.s_net.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='Valid')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output, _, _ = self.d_net(image)

            loss = self.criterion(output, target)
            if loss.dim() != 0:
                loss = loss.mean()
            test_loss += loss.item()
            tbar.set_description(f'Val loss: {test_loss/(i+1):.3f}')

            preds = output.detach().cpu().numpy().argmax(axis=1)
            self.evaluator.add_batch(target.cpu().numpy(), preds)

        Acc = self.evaluator.Pixel_Accuracy()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        fwIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print(f'Val @Epoch {epoch}: Acc={Acc:.4f}, mIoU={mIoU:.4f}, fwIoU={fwIoU:.4f}')

        if mIoU > self.best_pred:
            self.best_pred = mIoU
            self.saver.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': self.s_net.module.state_dict(),
                'best_pred': self.best_pred
            }, is_best=True)


def main():
    parser = argparse.ArgumentParser(description="KD Training for CropWeed")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet18','mobilenet','xception'], help='student backbone')
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='cropweed',
                        choices=['cropweed'], help='dataset name')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-size', type=int, default=1024)
    parser.add_argument('--crop-size', type=int, default=1024)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--freeze-bn', action='store_true', default=False)
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce','focal','kd'], help='loss func')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--use-balanced-weights', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly','step','cos'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ft', action='store_true', default=False)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-val', action='store_true', default=False)
    parser.add_argument('--pi_lambda', type=float, default=1.0)
    parser.add_argument('--pa_lambda', type=float, default=1.0)
    parser.add_argument('--teacher-weights', type=str,
                        default='pretrained/deeplab-resnet101.pth.tar',
                        help='path to frozen DeepLabV3+ ResNet-101 teacher')
    parser.add_argument('--checkname', type=str, default=None,
                        help='checkpoint directory name')

    args = parser.parse_args()
    # default checkpoint name
    if args.checkname is None:
        args.checkname = f'kd_{args.dataset}_{args.backbone}'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        
    if args.epochs is None:
        epoches = {
            'cropweed': 30,
            
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 6 * len(args.gpu_ids)

    # if args.test_batch_size is None:
    #     args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'cropweed': 0.01
        }
        args.lr = lrs[args.dataset.lower()] / (6 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epochs:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not args.no_val and epoch % args.eval_interval == args.eval_interval-1:
            trainer.validation(epoch)

if __name__ == '__main__':
    main()
