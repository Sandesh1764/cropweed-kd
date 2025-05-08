# Experiment summary (validation mIoU)

| Setting | Backbone | Epochs | Crop | mIoU |
|---------|----------|-------:|-----:|-----:|
| Scratch | MobileNetV2 | 30 | 513 | 0.7965 |
| Scratch | ResNet-18 | 30 | 513 | 0.7929 |
| KD (focal ablations) | MobileNetV2 | — | 513 | ≤0.712 |
| KD | MobileNetV2 | 80 | 513 | 0.7296 |
| **KD (best)** | **ResNet-18** | **150** | **1024** | **0.8652** |

Best checkpoint family: `kd_cropweed_resnet18/experiment_6`.
