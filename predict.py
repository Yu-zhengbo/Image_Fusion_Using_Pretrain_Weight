import torch
from torch import nn
from utils import GeneratorLoss, FusionDataset
from models import Generator
import os
import matplotlib.pyplot as plt
from torchvision.transforms import (
    Compose,
    RandomCrop,
    ToTensor,
    ToPILImage,
    CenterCrop,
    Resize,
    Grayscale,
)
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    vis_path = 'M3FD_00471_vis.png'
    ir_path = 'M3FD_00471_ir.png'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator()
    model.load_state_dict(
        torch.load("./logs/last.pth",map_location=device)
    )
    model.to(device)
    model.eval()

    vis_transform = Compose(
        [
            # Resize((512, 512)),
            # RandomCrop((224,224)),
            ToTensor(),
        ]
    )

    inf_transform = Compose([Grayscale(3), ToTensor()])

    v = Image.open(vis_path).convert("RGB")
    i = Image.open(ir_path).convert("L")
    v = vis_transform(v)
    i = inf_transform(i)
    v = v.unsqueeze(0).to(device)
    i = i.unsqueeze(0).to(device)
    # input = torch.cat((v,i),dim=1)

    fusion = model(v, i)

    plt.subplot(131)
    plt.imshow(v.squeeze(0).permute(1, 2, 0).cpu().numpy()[:,:,0],cmap ='gray')
    plt.subplot(132)
    plt.imshow(i.squeeze(0).permute(1, 2, 0).cpu().numpy()[:,:,0],cmap ='gray')
    plt.subplot(133)
    plt.imshow(fusion.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[:,:,0],cmap ='gray')
    plt.savefig('fusion.jpg',dpi=500)

    Image.fromarray(np.uint8(fusion.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[:,:,0]*255.)).save('fusion.png')
