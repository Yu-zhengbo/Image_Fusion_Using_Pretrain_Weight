import torch
from torch import nn
from utils import GeneratorLoss, FusionDataset
from models import Generator
import os
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

# parser = argparse.ArgumentParser(description="Config some basic settings.")
# parser.add_argument("--epoch", type=int, default=100)
# parser.add_argument("--batch_size", type=int, default=48)
# parser.add_argument("--num_workers", type=int, default=8)
# parser.add_argument("--fp16", type=bool, default=True)
# parser.add_argument('--local_rank', type=int, default=1, help='Local rank for distributed training')


if __name__ == "__main__":

    # args = parser.parse_args()
    # BATCH_SIZE = args.batch_size
    # AMP = args.fp16
    # EPOCH = args.epoch
    # NUM_WORKERS = args.num_workers

    BATCH_SIZE = 2
    AMP = False
    EPOCH = 100
    NUM_WORKERS = 2
    LOCAL_RANK = -1
    base_path = "/data/models/patent/datasets/M3FD/M3FD_Fusion"
    # base_path = "./M3FD_with_det"
    
    if LOCAL_RANK != -1:
        torch.distributed.init_process_group(
            backend="nccl"  # if torch.distributed.is_nccl_available() else "gloo"
        )
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", LOCAL_RANK)
        WORLD_SIZE = torch.cuda.device_count()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        WORLD_SIZE = 1

    
    train_dataset = FusionDataset(base_path,img_size=512)
    if LOCAL_RANK != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,num_workers=8, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
    epoch_step_train = len(train_dataset) // BATCH_SIZE // WORLD_SIZE

    model = Generator()
    model = model.to(device)
    model.train()
    
    if LOCAL_RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.92)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=EPOCH
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP))

    gen_loss = GeneratorLoss().to(device)
    loss_plot = []
    best_loss = 1000

    for _ in range(EPOCH):
        loss_epoch = 0
        if LOCAL_RANK != -1:
            train_loader.sampler.set_epoch(_)
        if LOCAL_RANK in [-1,0]:
            pbar_train = tqdm(
                total=epoch_step_train,
                desc=f"Epoch {_ + 1}/{EPOCH}",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                postfix=dict,
                mininterval=0.3,
            )
        for q, (vis, inf) in enumerate(train_loader):
            if q >= epoch_step_train:
                break
            optimizer.zero_grad()
            
            vis = vis.to(device)
            inf = inf.to(device)

            with torch.cuda.amp.autocast(enabled=(AMP)):
                fusion = model(vis, inf)
                loss = gen_loss(fusion, vis, inf)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_epoch += loss

            if LOCAL_RANK in [-1, 0]:
                pbar_train.set_postfix(
                    **{
                        "loss": loss_epoch.item() / (q + 1),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
                pbar_train.update(1)

        loss_epoch /= q + 1
        if LOCAL_RANK != -1:
            torch.distributed.barrier()

        lr_scheduler.step()


        # if LOCAL_RANK == 0:
        #     torch.distributed.reduce(
        #         loss_epoch, dst=0, op=torch.distributed.ReduceOp.SUM
        #     )
        #     loss_epoch /= WORLD_SIZE

        if LOCAL_RANK in [-1, 0]:
            if loss_epoch.item() < best_loss:
                best_loss = loss_epoch.item()
                torch.save(
                    model.state_dict()
                    if LOCAL_RANK == -1
                    else model.module.state_dict(),
                    "./logs/best.pth",
                )
                print('\nsave best model!\n')

            torch.save(
                model.state_dict()
                if LOCAL_RANK == -1
                else model.module.state_dict(),
                "./logs/last.pth",
            )

        loss_plot.append(loss_epoch.item())
        plt.figure()
        plt.plot(loss_plot)
        plt.savefig("loss.png")
        plt.close()
    
    if WORLD_SIZE > 1 and LOCAL_RANK == 0:
        torch.distributed.destroy_process_group()