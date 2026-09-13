# TFLite conversion and INT8 accuracy audit (Hybrid 18-class models)

Host evaluation (x86 CPU, TFLite 2.15 interpreter). All TFLite rows and the PIL reference share one preprocessing: PIL resize to 224x224 (bicubic) -> /255 -> ImageNet normalization (or [0,1] for graphs with embedded normalization). The canonical PyTorch evaluation transform (torchvision Resize, bilinear with antialias) is shown for comparison; the 1-2 pp gap between the two PyTorch rows is preprocessing, not conversion. `agree` = top-1 agreement with the PIL reference on the same images.

## ResNet-50

| Split | Variant | macro-F1 | acc | Δ F1 vs PyTorch | agree |
|---|---|---:|---:|---:|---:|
| PlantVillage test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.9685 | 0.9705 | -0.0051 | 0.986 |
| PlantVillage test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.9736 | 0.9773 | +0.0000 | 1.000 |
| PlantVillage test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.9463 | 0.9557 | -0.0273 | 0.963 |
| PlantVillage test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.9463 | 0.9557 | -0.0273 | 0.963 |
| PlantVillage test | onnx2tf float32 | 0.9736 | 0.9773 | +0.0000 | 1.000 |
| PlantVillage test | onnx2tf INT8 full-integer | 0.9714 | 0.9757 | -0.0022 | 0.997 |
| PlantDoc test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.1539 | 0.2037 | +0.0057 | 0.804 |
| PlantDoc test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.1482 | 0.2011 | +0.0000 | 1.000 |
| PlantDoc test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.1282 | 0.1849 | -0.0199 | 0.607 |
| PlantDoc test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.1284 | 0.1854 | -0.0198 | 0.607 |
| PlantDoc test | onnx2tf float32 | 0.1482 | 0.2011 | +0.0000 | 0.999 |
| PlantDoc test | onnx2tf INT8 full-integer | 0.1493 | 0.2021 | +0.0011 | 0.965 |

## MobileNetV1

| Split | Variant | macro-F1 | acc | Δ F1 vs PyTorch | agree |
|---|---|---:|---:|---:|---:|
| PlantVillage test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.9907 | 0.9935 | +0.0017 | 0.998 |
| PlantVillage test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.9889 | 0.9924 | +0.0000 | 1.000 |
| PlantVillage test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.1019 | 0.1775 | -0.8870 | 0.178 |
| PlantVillage test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.0043 | 0.0403 | -0.9846 | 0.041 |
| PlantVillage test | onnx2tf float32 | 0.9889 | 0.9924 | +0.0000 | 1.000 |
| PlantVillage test | onnx2tf INT8 full-integer | 0.9887 | 0.9916 | -0.0002 | 0.998 |
| PlantDoc test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.1270 | 0.1742 | +0.0025 | 0.878 |
| PlantDoc test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.1244 | 0.1742 | +0.0000 | 1.000 |
| PlantDoc test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.0288 | 0.1001 | -0.0957 | 0.372 |
| PlantDoc test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.0094 | 0.0919 | -0.1151 | 0.129 |
| PlantDoc test | onnx2tf float32 | 0.1244 | 0.1742 | +0.0000 | 1.000 |
| PlantDoc test | onnx2tf INT8 full-integer | 0.1218 | 0.1686 | -0.0026 | 0.923 |

## MobileNetV2

| Split | Variant | macro-F1 | acc | Δ F1 vs PyTorch | agree |
|---|---|---:|---:|---:|---:|
| PlantVillage test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.9833 | 0.9868 | +0.0073 | 0.989 |
| PlantVillage test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.9760 | 0.9800 | +0.0000 | 1.000 |
| PlantVillage test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.0043 | 0.0403 | -0.9717 | 0.038 |
| PlantVillage test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.0043 | 0.0403 | -0.9717 | 0.038 |
| PlantVillage test | onnx2tf float32 | 0.9760 | 0.9800 | +0.0000 | 1.000 |
| PlantVillage test | onnx2tf INT8 full-integer | 0.9762 | 0.9800 | +0.0002 | 0.998 |
| PlantDoc test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.1428 | 0.1656 | +0.0084 | 0.840 |
| PlantDoc test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.1344 | 0.1610 | +0.0000 | 1.000 |
| PlantDoc test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.0094 | 0.0924 | -0.1250 | 0.064 |
| PlantDoc test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.0094 | 0.0924 | -0.1250 | 0.064 |
| PlantDoc test | onnx2tf float32 | 0.1344 | 0.1610 | +0.0000 | 0.999 |
| PlantDoc test | onnx2tf INT8 full-integer | 0.1423 | 0.1686 | +0.0078 | 0.924 |

## EfficientNet-Lite0

| Split | Variant | macro-F1 | acc | Δ F1 vs PyTorch | agree |
|---|---|---:|---:|---:|---:|
| PlantVillage test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.9855 | 0.9900 | +0.0031 | 0.992 |
| PlantVillage test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.9824 | 0.9878 | +0.0000 | 1.000 |
| PlantVillage test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.9900 | 0.9932 | +0.0076 | 0.988 |
| PlantVillage test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.9815 | 0.9868 | -0.0009 | 0.996 |
| PlantDoc test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.2232 | 0.2793 | +0.0110 | 0.833 |
| PlantDoc test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.2122 | 0.2656 | +0.0000 | 1.000 |
| PlantDoc test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.1953 | 0.2377 | -0.0169 | 0.449 |
| PlantDoc test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.2053 | 0.2575 | -0.0069 | 0.928 |

## EfficientNet-B3

| Split | Variant | macro-F1 | acc | Δ F1 vs PyTorch | agree |
|---|---|---:|---:|---:|---:|
| PlantVillage test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.9850 | 0.9905 | +0.0013 | 0.997 |
| PlantVillage test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.9837 | 0.9897 | +0.0000 | 1.000 |
| PlantVillage test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.0080 | 0.0775 | -0.9757 | 0.077 |
| PlantVillage test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.0080 | 0.0775 | -0.9757 | 0.077 |
| PlantVillage test | onnx2tf float32 | 0.9837 | 0.9897 | +0.0000 | 1.000 |
| PlantVillage test | onnx2tf INT8 full-integer | 0.9582 | 0.9676 | -0.0255 | 0.973 |
| PlantDoc test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.2553 | 0.2880 | +0.0002 | 0.892 |
| PlantDoc test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.2551 | 0.2890 | +0.0000 | 1.000 |
| PlantDoc test | legacy Keras-rebuild INT8 (2026-02-08 file) | 0.0054 | 0.0513 | -0.2497 | 0.055 |
| PlantDoc test | Keras-rebuild INT8, current checkpoint (10.x scripts) | 0.0054 | 0.0513 | -0.2497 | 0.055 |
| PlantDoc test | onnx2tf float32 | 0.2551 | 0.2890 | -0.0000 | 0.999 |
| PlantDoc test | onnx2tf INT8 full-integer | 0.1456 | 0.1823 | -0.1096 | 0.434 |

## ResNet-50 DANN seed 42

| Split | Variant | macro-F1 | acc | Δ F1 vs PyTorch | agree |
|---|---|---:|---:|---:|---:|
| PlantVillage test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.9465 | 0.9503 | -0.0145 | 0.980 |
| PlantVillage test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.9611 | 0.9660 | +0.0000 | 1.000 |
| PlantVillage test | Keras-rebuild INT8 (10.4 path via 14.3, 2026-09-12) | 0.7864 | 0.8290 | -0.1747 | 0.845 |
| PlantVillage test | onnx2tf float32 | 0.9611 | 0.9660 | +0.0000 | 1.000 |
| PlantVillage test | onnx2tf INT8 full-integer | 0.9600 | 0.9651 | -0.0011 | 0.997 |
| PlantDoc test | PyTorch FP32, torchvision Resize (canonical eval transform) | 0.2402 | 0.2834 | -0.0086 | 0.861 |
| PlantDoc test | PyTorch FP32, PIL preprocessing (reference for TFLite rows) | 0.2487 | 0.2920 | +0.0000 | 1.000 |
| PlantDoc test | Keras-rebuild INT8 (10.4 path via 14.3, 2026-09-12) | 0.1840 | 0.2458 | -0.0648 | 0.583 |
| PlantDoc test | onnx2tf float32 | 0.2487 | 0.2920 | -0.0000 | 0.999 |
| PlantDoc test | onnx2tf INT8 full-integer | 0.2468 | 0.2920 | -0.0019 | 0.967 |
