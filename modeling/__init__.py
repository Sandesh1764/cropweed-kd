def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet18':
        from modeling.backbone.resnet import ResNet18
        return ResNet18(layers=[2, 2, 2, 2], output_stride=output_stride, BatchNorm=BatchNorm)
    else:
        raise NotImplementedError(f'Backbone {backbone} not implemented')
def ResNet18(output_stride, BatchNorm):
    from modeling.backbone.resnet import ResNet18
    return ResNet18(layers=[2, 2, 2, 2], output_stride=output_stride, BatchNorm=BatchNorm, architecture = 'resnet18')