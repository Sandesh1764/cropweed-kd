# CropWeed data layout

```text
data/cropweed/
├── images/
│   ├── train/*.png
│   └── val/*.png
└── labels/
    ├── train/*.png   # 0=soil, 1=crop, 2=weed
    └── val/*.png
```

Set `CROPWEED_ROOT` to this folder, or place it at `data/cropweed`.

Teacher DeepLabV3+ ResNet-101 weights go in `pretrained/` (see main README).
