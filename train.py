import torch
from torch import nn
from utils import GeneratorLoss,FusionDataset
from models import Generator
import os
import matplotlib.pyplot as plt



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    base_path = '/data/models/patent/datasets/M3FD/M3FD_Fusion'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = FusionDataset(base_path)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True,drop_last=True)

    model = Generator()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay = 5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.92)
    
    gen_loss = GeneratorLoss()
    loss_plot =  []
    for _ in range(100):
        loss_epoch = 0
        for q,(vis,inf) in enumerate(train_loader):
            # if q == 10:
            #     break
            vis = vis.to(device)
            inf = inf.to(device)
            fusion = model(vis,inf)
            loss = gen_loss(fusion, vis, inf)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        
        if _ % 10 == 0:
            print(f'epoch: {_}, loss: {loss_epoch/q}')

        loss_plot.append(loss.item()/q)
        lr_scheduler.step()

    torch.save(model.state_dict(), 'model.pth')
    plt.figure()
    plt.plot(loss_plot)
    plt.savefig('loss.png')