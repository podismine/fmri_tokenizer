import torch
import os

import torch.nn as nn
from data import Task1Data
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from models.mine_cvq import VQVAE

log_dir = "logs/log_cvqvae_mamba_con"
checkpoint_dir = "checkpoint/checkpoint_cvqvae_mamba_con"

def try_do(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Error: {e}")

resume = False
use_model = 'mamba'
if resume is not True:
    try_do(shutil.rmtree, log_dir)
    try_do(os.makedirs, log_dir)
    try_do(shutil.rmtree, checkpoint_dir)
    try_do(os.makedirs, checkpoint_dir)

writer = SummaryWriter(log_dir=log_dir)
train_dataset = Task1Data(is_train=True)
train_loader = DataLoader(train_dataset, batch_size=256,num_workers=4)
model = VQVAE(input_dim = 100, hidden_dim = 512, state_num_embeddings = 512, transition_num_embeddings = 16, embedding_dim = 256, commitment_cost = 0.1, model = use_model) # 
if resume is True:
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'model_7000.pth'), map_location = 'cpu')
    model.load_state_dict(checkpoint['model'])
model.cuda().train()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

optimizer_model = torch.optim.AdamW(model.parameters(),lr= 5e-4,weight_decay=0.0001)

best_train_loss = 99999.

def save_model(model, name):
    torch.save({
            'model':model.state_dict(),
            }, os.path.join(f"{checkpoint_dir}/{name}.pth"))
    
for epoch_counter in range(15000):
    total_train_loss = 0.
    train_rec_loss = 0.
    train_vq_loss_state = 0.
    train_vq_loss_transision = 0.
    count = 0.

    model.train()
    for step, (x, label) in enumerate(train_loader):
        # padded_tensor = F.pad(tensor, (0, 2))
        x = x.to(device,non_blocking=True).float()#[:,0].permute(0,2,1)
        # label = label.to(device).long()

        optimizer_model.zero_grad()
        reconstructed, vq_loss_state, vq_loss_transition = model(x)
        
        recon_loss = nn.MSELoss()(reconstructed, x)
        loss = recon_loss + vq_loss_state + vq_loss_transition

        loss.backward()
        optimizer_model.step()

        total_train_loss += len(x) * float(loss)
        train_rec_loss += len(x) * float(recon_loss)
        train_vq_loss_state += len(x) * float(vq_loss_state)
        train_vq_loss_transision += len(x) * float(vq_loss_transition)
        count += len(x)


    total_train_loss /= count
    train_rec_loss /= count
    train_vq_loss_state /= count
    train_vq_loss_transision /= count

    print(f"[{epoch_counter}]: train_loss: {total_train_loss:.4f}")
    print(f"[{epoch_counter}]: train_rec_loss: {train_rec_loss:.4f}")
    print(f"[{epoch_counter}]: train_vq_loss_state: {train_vq_loss_state:.4f}")
    print(f"[{epoch_counter}]: train_vq_loss_transision: {train_vq_loss_transision:.4f}")
    print("*"*40)

    writer.add_scalar('train_loss', total_train_loss, global_step=epoch_counter)
    writer.add_scalar('train_rec_loss', train_rec_loss, global_step=epoch_counter)
    writer.add_scalar('train_vq_loss_state', train_vq_loss_state, global_step=epoch_counter)
    writer.add_scalar('train_vq_loss_transision', train_vq_loss_transision, global_step=epoch_counter)

    if total_train_loss < best_train_loss:
        best_train_loss = total_train_loss
        save_model(model, 'best_model')
    if epoch_counter % 1000 == 0:
        save_model(model, f'model_{epoch_counter}')
    save_model(model, 'last_model')