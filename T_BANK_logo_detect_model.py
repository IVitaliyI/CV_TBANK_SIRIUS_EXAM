import torch
import torchvision.transforms.v2 as tfs
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import tqdm

train_transform = tfs.Compose([
    tfs.Resize((160, 160)),
    tfs.RandomResizedCrop(128, scale=(0.7, 1.0)),
    tfs.RandomHorizontalFlip(),
    tfs.RandomRotation(15),
    tfs.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    tfs.RandomApply([tfs.GaussianBlur(3)], p=0.2),
    tfs.ToTensor(),
    tfs.ToDtype(torch.float32, scale=True),
])

val_transform = tfs.Compose([
    tfs.Resize((128, 128)),
    tfs.ToTensor(),
    tfs.ToDtype(dtype=torch.float32, scale=True)
])

dataset = ImageFolder(r"data/Logo/train", transform=train_transform)

train_loader: DataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

epochs = 200

arr = tqdm.tqdm(range(epochs))

for epoch in arr:
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
