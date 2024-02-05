import torch.nn as nn
import torch
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, model_urls
import copy

class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


def _resnet(arch, block, layers, pretrained, progress, channel=3, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if channel==1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def get_model(pretrain=True):
    backbone = resnet50(pretrained=pretrain)
    return backbone


class ImageClassifierHead(nn.Module):
  
    def __init__(self, in_features: int, pool_layer=None, num_classes=1):
        super(ImageClassifierHead, self).__init__()
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.Identity()
        self.head  = nn.Linear(in_features, num_classes)
            

    def forward(self, inputs: torch.Tensor, get_f=False) -> torch.Tensor:
        f = self.pool_layer(inputs)
        if get_f:
            return torch.squeeze(self.head(f),dim=1), f
        else:
            return torch.squeeze(self.head(f),dim=1)