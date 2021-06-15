import torch
import torch.nn as nn
from typing import Any
from .cam import GradCAM
from libs.fix_weight_dict import fix_model_state_dict

class BENet(nn.Module):

    def __init__(self, pretrained: bool = False, in_channels: int = 3, num_classes: int = 3) -> None:
        super(BENet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.global_maxpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_maxpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def benet(pretrained:bool=False, **kwargs: Any) -> BENet:
    model = BENet(**kwargs)
    if pretrained:
        state_dict = torch.load('./pretrained/pretrained_benet.prm')#map_location
        model.load_state_dict(fix_model_state_dict(state_dict))
    return model

def cam_benet(pretrained:bool=False, **kwargs: Any) -> BENet:
    model = BENet(**kwargs)
    state_dict = torch.load('./pretrained/pretrained_benet.prm')#map_location
    model.load_state_dict(fix_model_state_dict(state_dict))
    model.eval()
    target_layer = model.features[3]
    wrapped_model = GradCAM(model, target_layer)
    return wrapped_model
