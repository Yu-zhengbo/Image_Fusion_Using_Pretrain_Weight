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
import copy
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    base_path = "./M3FD_with_det/vis"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator()
    model.load_state_dict(
        torch.load("./logs/last.pth",map_location=device)
    )
    model.to(device)
    model.eval()

    vis_transform = Compose(
        [
            # Resize((256, 256)),
            # RandomCrop((224,224)),
            ToTensor(),
        ]
    )

    inf_transform = Compose([Grayscale(3), ToTensor()])

    # for v in os.listdir(base_path):
    for v in tqdm(os.listdir(base_path), desc="Processing images"):
        v = os.path.join(base_path, v)
        name = copy.deepcopy(v)
        i = v.replace("vis", "ir")
        v = Image.open(v).convert("RGB")
        i = Image.open(i).convert("L")
        v = vis_transform(v)
        i = inf_transform(i)
        v = v.unsqueeze(0).to(device)
        i = i.unsqueeze(0).to(device)
        fusion = model(v, i)

        fusion = fusion.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255.
        fusion = Image.fromarray(fusion.astype("uint8"))

        fusion.save(name.replace("vis","fusion"))
