import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import PIL
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

# Custom dataset class
class CatDogDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path, label = self.file_paths[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

# Define the neural network architecture
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Adjust output to 2 classes (dogs and cats)

    def forward(self, x):
        return self.model(x)

# Load data
file_paths = []
for class_name in ["cats", "dogs"]:
    for img_path in glob.glob(f"./data/cats_and_dogs_filtered/train/{class_name}/*.jpg"):
        file_paths.append((img_path, 0 if class_name == "cats" else 1))

# Split data into train and validation sets
train_paths, val_paths = train_test_split(file_paths, test_size=0.2, random_state=42)

# Define transformations
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = CatDogDataset(train_paths, transform=preprocess)
val_dataset = CatDogDataset(val_paths, transform=preprocess)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = CatDogClassifier()

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define early stopping parameters
patience = 5
best_validation_loss = float('inf')
current_patience = 0

# Training loop with early stopping
for epoch in range(10):  # Limiting to 10 epochs for demonstration
    # Training steps
    model.train()
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation steps
    model.eval()
    with torch.no_grad():
        validation_loss = 0.0
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

    # Average validation loss
    validation_loss /= len(val_dataloader)

    # Check for improvement in validation loss
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        current_patience = 0
        # Save the model if desired
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        current_patience += 1
        # Check if early stopping criteria are met
        if current_patience > patience:
            print("Early stopping! No improvement for {} epochs.".format(patience))
            break
