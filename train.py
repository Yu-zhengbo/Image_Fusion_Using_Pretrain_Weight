import torch
from torch import nn
from utils import GeneratorLoss,FusionDataset
from models import Generator


if __name__ == "__main__":
    base_path = r'D:\迅雷下载\image_fusion\M3FD_Detection-001'

    train_dataset = FusionDataset(base_path)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=2,shuffle=True,drop_last=True)

    model = Generator()

    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay = 5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    
    gen_loss = GeneratorLoss()
    
    for _ in range(10):
        loss_epoch = 0
        for q,(vis,inf) in enumerate(train_loader):
            if q == 10:
                break
            fusion = model(vis,inf)
            loss = gen_loss(fusion, vis, inf)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            
        print(loss_epoch/10)
        lr_scheduler.step()
        