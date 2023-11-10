import torch
from torch import nn
from torchvision.models.vgg import vgg16,vgg13
import math
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import ToPILImage
import random
from torch.autograd import Variable

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.local_feature = local_feature_information()
        self.globa_feature = global_feature_information()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.laplace = LaplacianLoss()
        self.blur_kernel = get_gaussian_kernel(kernel_size=3)
        self.ssim = SSIM()

    def get_salient_feature(self,x,delta):
        B,_,H,W = x.shape
        x = torch.mean(x,dim=1,keepdim=True)>delta
        return x


    def forward(self, fusion, visible, infrared):

        # B,C,H,W = fusion.shape
        
        vis_loss = 0
        inf_loss = 0

        for feature_model in [self.local_feature,self.globa_feature]:
            for f,v,i in zip(feature_model(fusion),feature_model(visible),feature_model(infrared)):
                #salient区域直接计算感知损失，非salient区域计算非感知的fusion和vis或inf损失。
                salient_vis = self.get_salient_feature(v,0.7)
                salient_ir = self.get_salient_feature(i,0.3)

                # v_grad = self.laplace(v)
                # i_grad = self.laplace(i)
                # f_grad = self.laplace(f)

                v_grad = self.blur_kernel(self.laplace(v))
                i_grad = self.blur_kernel(self.laplace(i))
                f_grad = self.blur_kernel(self.laplace(f))

                vis_loss += self.mse_loss(f*salient_vis,v*salient_vis)
                vis_loss += self.mse_loss(~salient_vis*f_grad,~salient_vis*v_grad)

                inf_loss += self.mse_loss(f*salient_ir,i*salient_ir)
                inf_loss += self.mse_loss(~salient_ir*f_grad,~salient_ir*i_grad)

        ssim_loss = (1-self.ssim(fusion,visible)) + (1-self.ssim(fusion,infrared))
        tv_loss = self.tv_loss(fusion)

        return vis_loss+inf_loss+ssim_loss+tv_loss

def get_gaussian_kernel(kernel_size=5, sigma=5, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter
    
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class LaplacianLoss(nn.Module):
    def __init__(self,channels=3):
        super(LaplacianLoss, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        return self.laplacian_filter(x) ** 2
    

class local_feature_information(nn.Module):
    def __init__(self):
        super(local_feature_information, self).__init__()
        from torchvision.models.resnet import resnet18
        resnet = resnet18(pretrained=True).eval()
        for param in resnet.parameters():
            param.requires_grad = False

        self.conv_bn_relu_maxpool = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        del resnet
        torch.cuda.empty_cache()

    def forward(self,x):
        x = self.conv_bn_relu_maxpool(x)
        x1 = self.layer1(x)
        x2 =self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1,x2,x3,x4
    

class global_feature_information(nn.Module):
    def __init__(self):
        super(global_feature_information, self).__init__()
        from torchvision.models.swin_transformer import swin_t
        swin = swin_t(pretrained=True).eval()
        for param in swin.parameters():
            param.requires_grad = False

        self.layer1 = nn.Sequential(*list(swin.features)[:2])
        self.layer2 = nn.Sequential(*list(swin.features)[2:4])
        self.layer3 = nn.Sequential(*list(swin.features)[4:6])
        self.layer4 = nn.Sequential(*list(swin.features)[6:8])

        del swin
        torch.cuda.empty_cache()

    def forward(self,x):
        x1 = self.layer1(x)
        x2 =self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1.permute(0,3,1,2),x2.permute(0,3,1,2),x3.permute(0,3,1,2),x4.permute(0,3,1,2)


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.to(img1.device)
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def create_window(self,window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window


    def _ssim(self,img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


    def gaussian(self,window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

if __name__ == "__main__":
    model =  global_feature_information()
    input = torch.randn(2,3,640,640)
    for o in model(input):
        print(o.shape)

    model =  local_feature_information()
    input = torch.randn(2,3,640,640)
    for o in model(input):
        print(o.shape)

    img1 = Variable(torch.rand(1, 3, 256, 256))
    img2 = Variable(torch.rand(1, 3, 256, 256))
    ssim_loss = SSIM()
    loss = ssim_loss(img1, img2)
    print(loss)