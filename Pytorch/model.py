import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from dataLoader import DanceDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
lr = 1e-3
batch_size = 16
epochs = 10

trans = transform.Compose([transform.Resize((224,224)),transform.ToTensor()])
dataset = DanceDataset(csv_file= "E:\\Aquib\\MCA\\Python\\Dance Form Detection\\Pytorch\\torch_train.csv",
root_dir = "E:\\Aquib\\MCA\\Python\\Dance Form Detection\\dataset\\train\\",
transform = trans
)

train_set,test_set = torch.utils.data.random_split(dataset,[292,72])
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True)

model = torchvision.models.densenet121(pretrained=True)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):
    losses = []
    for batch_idx,(data,targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    print(f"Cost at Epoch {epoch} is {sum(losses)/len(losses)}")
