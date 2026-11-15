"""ResNet18 baseline and controlled CBAM placements."""
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
from .cbam import CBAM

VARIANTS = ("baseline", "cbam_before_add", "cbam_after_add", "cbam_stage_end")

class AttentionBasicBlock(BasicBlock):
    """CBAM before residual sum, or after sum but before final ReLU.

    The residual branch and skip branch remain unchanged. All attention
    variants start from scratch with the same small-image stem.
    """
    def __init__(self, *args, placement: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.placement = placement
        self.cbam = CBAM(self.bn2.num_features)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.placement == "before_add":
            out = self.cbam(out)
        out = out + identity
        if self.placement == "after_add":
            out = self.cbam(out)
        return self.relu(out)

def create_resnet18(num_classes: int, *, variant: str = "baseline", small_image_stem: bool = True) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    model = resnet18(weights=None, num_classes=num_classes)
    if small_image_stem:
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    if variant in ("cbam_before_add", "cbam_after_add"):
        placement = "before_add" if variant == "cbam_before_add" else "after_add"
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(model, layer_name)
            for i, old in enumerate(layer):
                new = AttentionBasicBlock(old.conv1.in_channels, old.conv2.out_channels,
                    stride=old.conv1.stride[0], downsample=old.downsample,
                    groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d,
                    placement=placement)
                layer[i] = new
    elif variant == "cbam_stage_end":
        for name, channels in zip(("layer1", "layer2", "layer3", "layer4"), (64, 128, 256, 512)):
            setattr(model, name, nn.Sequential(getattr(model, name), CBAM(channels)))
    return model
