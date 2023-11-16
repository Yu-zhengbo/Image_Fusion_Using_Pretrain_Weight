import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    r"""
    Use to generate fused images.
    ir + vi -> fus
    """

    def __init__(self, dim: int = 32, depth: int = 3):
        super(Generator, self).__init__()
        self.depth = depth

        self.encoder = nn.Sequential(
            nn.Conv2d(6, dim, (3, 3), (1, 1), 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.dense = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * (i + 1), dim, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ) for i in range(depth)
        ])

        self.fuse = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(dim * (depth + 1), dim * 4, (3, 3), (1, 1), 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 4, dim * 2, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim * 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim, 3, (3, 3), (1, 1), 1),
                nn.Tanh()
            ),
        )

    def forward(self, vis: Tensor, ir: Tensor) -> Tensor:
        src = torch.cat([vis, ir], dim=1)
        _,_,h,w = src.shape
        x = self.encoder(src)
        for i in range(self.depth):
            t = self.dense[i](x)
            x = torch.cat([x, t], dim=1)
        fus = self.fuse(x)
        if fus.shape[2] != h or fus.shape[3] != w:
            fus = nn.functional.interpolate(fus, size=(h, w), mode='bilinear')
        return nn.Sigmoid()(fus)


if __name__ == "__main__":
    img = torch.randn(2,3,768,1024)
    # gen_vis = Generator()
    # gen_inf = Generator()
    # fus_vis = gen_vis(img)
    # print(fus_vis.shape)
    # from torchvision.models.vgg import vgg16,vgg13
    # from torchvision.models.resnet import resnet18
    # from torchvision.models.swin_transformer import swin_t


    # resnet = resnet18(pretrained=True)
    # swin = swin_t(pretrained=True)

    # print(resnet)

    # vgg = vgg16(pretrained=False)
    # loss_network = nn.Sequential(*list(vgg.features)[:25]).eval()
    # img = torch.randn(2,3,768,1024)
    # output = loss_network(img)
    # print(output.shape)