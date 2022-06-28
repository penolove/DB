import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from decoders import SegDetector


# ref: https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
def _resnet(
    block,
    layers,
    weights,
    progress,
    **kwargs,
):
    model = ResNet18FPN(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class ResNet18FPN(ResNet):
    def forward_fpn(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


class BasicModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = _resnet(BasicBlock, [2, 2, 2, 2], weights=None, progress=True)
        self.decoder = SegDetector(in_channels=[64, 128, 256, 512], adaptive=True, k=50)

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)
