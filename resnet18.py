import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

DATA_DIR = r"D:\univ\Deep Learning\ass\data"
BATCH_SIZE = 4  
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
   
    if not os.path.exists(DATA_DIR):
        print(f"Error: The path '{DATA_DIR}' does not exist.")
        print("Please check the folder name and try again.")
        return

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading data from:", DATA_DIR)
    
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                          for x in ['train', 'val']}
        
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
                       for x in ['train', 'val']}
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        
     
        num_classes = len(class_names)
        
        print(f"Success! Found {num_classes} classes: {class_names}")
        print(f"Training images: {dataset_sizes['train']}, Validation images: {dataset_sizes['val']}")

    except FileNotFoundError as e:
        print("\nError: Could not find 'train' or 'val' folders inside your data path.")
        print(f"Expected structure:\n{DATA_DIR}\\train\n{DATA_DIR}\\val")
        return

    print("\nLoading pre-trained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"\nStarting training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("\nTraining Complete.")
    
  
    save_path = os.path.join(DATA_DIR, 'my_finetuned_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == '__main__':
    main()