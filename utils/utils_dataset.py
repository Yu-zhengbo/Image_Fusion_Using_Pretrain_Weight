import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
from torchvision.transforms import functional as F


def resize_image(image, size, letterbox_image=True):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (0,0,0))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


class FusionDataset(Dataset):
    def __init__(self, root, is_train=True,with_det=False):
        self.root = root
        self.is_train = is_train
        self.visible_path = os.path.join(self.root, 'vis')
        self.infrared_path = os.path.join(self.root, 'ir')
        if with_det:
            self.ann_path = os.path.join(self.root, 'ann')

        self.visible_img_list = os.listdir(self.visible_path)


        self.vis_transform = Compose([
            Resize((256,256)),
            # RandomCrop((224,224)),
            ToTensor()
            ])

        self.inf_transform = Compose([
            Grayscale(3),
            Resize((256,256)),
            ToTensor()
            ])


    def __getitem__(self, index):

        vis = Image.open(os.path.join(self.visible_path, self.visible_img_list[index])).convert('RGB')
        inf = Image.open(os.path.join(self.infrared_path, self.visible_img_list[index])).convert('L')

        vis = self.vis_transform(vis)
        inf = self.inf_transform(inf)

        if self.is_train:
            inf_mask = torch.rand((inf.shape[1],inf.shape[2]))
            inf[:,inf_mask<0.05] = 0
        return vis,inf

    def __len__(self):
        return len(self.visible_img_list)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = FusionDataset(root=r'/data/models/patent/roadscene')
    vis, inf = dataset[0]
    plt.subplot(121)
    plt.imshow(vis.permute(1,2,0))
    plt.subplot(122)    
    plt.imshow(inf.permute(1,2,0))
    plt.show()
    