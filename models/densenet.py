import torch
from torchvision import models


def densenet121(channels, out_features, use_gpu, pretrained, drop_rate):
    model = models.densenet121(pretrained=pretrained, drop_rate=drop_rate)

    for parma in model.parameters():
        parma.requires_grad = False
    model.features.conv0 = torch.nn.Sequential(
        torch.nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=1024, out_features=out_features, bias=True))

    if use_gpu:
        model = model.cuda()

    return model
