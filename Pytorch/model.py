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

model = torchvision.models.vgg16_bn(pretrained=True)
# print(model)
for param in model.parameters():
    param.requires_grad= False

model.classifier[6] = nn.Sequential(nn.Linear(4096,8))

# print(model.parameters)
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
    print(f"Loss at Epoch {epoch} is {loss}")

correct = 0
total = 0
with torch.no_grad():
    for images,labels in test_loader:
        # images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))