#%%
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp


#%%
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = VOCSegmentation(root, year='2012', image_set='train', download=True, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(target, dtype=torch.long)
        return image, target

# Hyperparameters
num_classes = 2
learning_rate = 0.001
batch_size = 4
num_epochs = 10

# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load dataset
train_dataset = CustomDataset(root='./data', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, targets in train_loader:
        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")
