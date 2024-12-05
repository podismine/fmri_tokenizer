import torch
import os

import torch.nn as nn
from data import Task1Data
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from models.mine_cvq import VQVAE
from utils import log_util, parse_args

args = parse_args()
resume = args.resume
use_model = args.model
args.log_name = f'{args.log_name}_{use_model}_st-{args.state_token}_tt-{args.trans_token}' # 4, 8, 16, 32

log_dir = f"/home/yyang/yang/fmri_tokenizer/logs/{args.log_name}" #"logs/log_cvqvae_mamba_con"
checkpoint_dir = f'/home/yyang/yang/fmri_tokenizer/checkpoint/{args.check_name}'#"checkpoint/checkpoint_cvqvae_mamba_con"

def try_do(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Error: {e}")

train_dataset = Task1Data(is_train=True)
val_dataset = Task1Data(is_train=False, is_test=False)

train_loader = DataLoader(train_dataset, batch_size=256,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256,num_workers=4)

model = VQVAE(input_dim = 100, hidden_dim = args.hidden, state_num_embeddings = args.state_token, transition_num_embeddings = args.trans_token, layer = args.layer, embedding_dim = 256, commitment_cost = 0.1, model = use_model) # 

if resume is not True:
    # try_do(shutil.rmtree, log_dir)
    try_do(os.makedirs, log_dir)
    # try_do(shutil.rmtree, checkpoint_dir)
    try_do(os.makedirs, checkpoint_dir)
else:
    checkpoint = torch.load(os.path.join(checkpoint_dir, args.resume_checkpoint), map_location = 'cpu')
    model.load_state_dict(checkpoint['model'])

model.cuda().train()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
optimizer_model = torch.optim.AdamW(model.parameters(),lr= 5e-4,weight_decay=0.0001)
writer = SummaryWriter(log_dir=log_dir)

def save_model(model, name):
    torch.save({'model':model.state_dict(),}, os.path.join(f"{checkpoint_dir}/{name}.pth"))

best_loss = 9999999.
for epoch_counter in range(args.epochs):

    log_train_total_loss = log_util("Training_total_loss", writer)
    log_train_recon_loss = log_util("Training_recon_loss", writer)
    log_train_vq_state_loss = log_util("Training_vq_state_loss", writer)
    log_train_vq_trans_loss = log_util("Training_vq_transit_loss", writer)

    log_val_total_loss = log_util("Val_total_loss", writer, is_train=False)
    log_val_recon_loss = log_util("Val_recon_loss", writer, is_train=False)
    log_val_vq_state_loss = log_util("Val_vq_state_loss", writer, is_train=False)
    log_val_vq_trans_loss = log_util("Val_vq_transit_loss", writer, is_train=False)

    model.train()
    for step, (x, label) in enumerate(train_loader):
        x = x.to(device,non_blocking=True).float()

        optimizer_model.zero_grad()
        reconstructed, vq_loss_state, vq_loss_transition = model(x)
        
        recon_loss = nn.MSELoss()(reconstructed, x)
        loss = recon_loss + vq_loss_state + vq_loss_transition

        loss.backward()
        optimizer_model.step()

        log_train_total_loss.update(loss, len(x))
        log_train_recon_loss.update(recon_loss, len(x))
        log_train_vq_state_loss.update(vq_loss_state, len(x))
        log_train_vq_trans_loss.update(vq_loss_transition, len(x))

    print("*"*40)
    log_train_total_loss.summary(epoch_counter)
    log_train_recon_loss.summary(epoch_counter)
    log_train_vq_state_loss.summary(epoch_counter)
    log_train_vq_trans_loss.summary(epoch_counter)

    ############################################## model eval #################################################
    if epoch_counter <= 5000:
        # continue
        pass

    model.eval()
    for step, (x, label) in enumerate(val_loader):
        x = x.to(device,non_blocking=True).float()
        with torch.no_grad():
            reconstructed, vq_loss_state, vq_loss_transition = model(x)
        
        recon_loss = nn.MSELoss()(reconstructed, x)
        loss = recon_loss + vq_loss_state + vq_loss_transition

        log_val_total_loss.update(loss, len(x))
        log_val_recon_loss.update(recon_loss, len(x))
        log_val_vq_state_loss.update(vq_loss_state, len(x))
        log_val_vq_trans_loss.update(vq_loss_transition, len(x))

    log_val_total_loss.summary(epoch_counter)
    log_val_recon_loss.summary(epoch_counter)
    log_val_vq_state_loss.summary(epoch_counter)
    log_val_vq_trans_loss.summary(epoch_counter)

    monior =  log_val_recon_loss.avg

    if monior < best_loss:
        best_loss = monior
        save_model(model, 'best_model')

    if epoch_counter % 1000 == 0:
        save_model(model, f'model_{epoch_counter}')

    save_model(model, 'last_model')