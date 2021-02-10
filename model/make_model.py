import torch
import torch.nn as nn
from model.backbones.ibnnet.resnext_ibn import resnext101_ibn_a


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        self.in_planes = 2048
        self.base = resnext101_ibn_a(pretrained=True)
        self.base = torch.nn.Sequential(*(list(self.base.children())[:-1]))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        # self.bottleneck = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck.bias.requires_grad_(False)
        # self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        x = x.view(x.shape[0], -1)  # flatten to (bs, 2048)
        score = self.classifier(x)
        return score

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
