import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, vit_b_16
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# -----------------------------
# Tiny CNN (FASTEST – pipeline test)
# -----------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # works for any image size
            nn.Flatten(),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# -----------------------------
# Custom CNN (still lightweight)
# -----------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),               # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),               # 16x16 -> 8x8
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -----------------------------
# Model factory
# -----------------------------
def make_model(
    arch="tinycnn",
    num_classes=10,
    pretrained=False,
    freeze_backbone=False
):
    arch = arch.lower()

    # 🚀 FAST MODE (recommended for CPU testing)
    if arch == "tinycnn":
        model = TinyCNN(num_classes)

    elif arch == "customcnn":
        model = CustomCNN(num_classes)

    # ⚠️ Heavy models (slow on CPU)
    elif arch == "convnext_tiny":
        model = convnext_tiny(weights="DEFAULT" if pretrained else None)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            num_classes
        )

    elif arch == "vit":
        model = vit_b_16(weights="DEFAULT" if pretrained else None)
        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes
        )

    elif arch == "fasterrcnn_backbone":
        backbone_model = fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None
        )

        model = nn.Sequential(
            backbone_model.backbone,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_model.backbone.out_channels, num_classes)
        )

    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # -----------------------------
    # Freeze backbone (safe)
    # -----------------------------
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not any(k in name for k in ["classifier", "head", "fc"]):
                param.requires_grad = False

    return model
