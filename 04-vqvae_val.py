import torch
import numpy as np

from data import Task1Data
from torch.utils.data.dataloader import DataLoader
from sklearn import preprocessing

from models.mine import VQVAE
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

train_dataset = Task1Data(is_train=True,is_test=False)
val_dataset = Task1Data(is_train=False,is_test=False)
test_dataset = Task1Data(is_train=False,is_test=True)

train_loader = DataLoader(train_dataset, batch_size=32,num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32,num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32,num_workers=0)

model = VQVAE(input_dim = 100, hidden_dim = 512, state_num_embeddings = 512, transition_num_embeddings = 16, embedding_dim = 256, commitment_cost = 0.1) # 
# model = VQVAE(input_dim = 100, hidden_dim = 512, num_embeddings = 512, embedding_dim = 256, commitment_cost = 0.1) # 

checkpoint = torch.load("checkpoint/checkpoint_vqvae_mine/last_model.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.cuda().eval()

def extract_embeddings(dataloader):
    x_stack = []
    y_stack = []
    for idx, (x, y) in enumerate(dataloader):
        x = x.cuda().float()#[:,0].permute(0,2,1)
        with torch.no_grad():
            tokens1, tokens2 = model.forward_token(x)
            x_stack.extend(tokens2.detach())
            y_stack.extend(y)
    x_stack = torch.stack(x_stack).detach().cpu().numpy()
    y_stack = torch.stack(y_stack).detach().cpu().numpy()
    # x_stack = x_stack.reshape(len(x_stack), -1)
    x_stack = x_stack.mean(1)
    # (863, 64, 256)
    return x_stack, y_stack


x_train, y_train = extract_embeddings(train_loader)
x_val, y_val = extract_embeddings(val_loader)
x_test, y_test = extract_embeddings(test_loader)


scaler = preprocessing.StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train).astype(np.float32)
x_val = scaler.transform(x_val).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)

# print(x_train.shape, y_train.shape)
# print(x_train[0])
clf = SVC(probability=True) # 5

clf.fit(x_train, y_train)
pred_test = clf.predict(x_test)

acc = accuracy_score(pred_test, y_test)
cm = confusion_matrix(pred_test, y_test)
sen = round(cm[1, 1] / float(cm[1, 1]+cm[1, 0]),4)
spe = round(cm[0, 0] / float(cm[0, 0]+cm[0, 1]),4)

res_string = f"acc: {acc:.4f}  sen: {sen:.4f} spe: {spe:.4f}"
print(res_string)