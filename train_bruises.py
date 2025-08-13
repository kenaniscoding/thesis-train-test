import torch, os, sys
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # Name of the first GPU
print("-----")
print(torch.version.cuda)  # CUDA version PyTorch was built with
print(torch.__version__)   # PyTorch version

# CHANGEME IMG DATASET DIR
data_dir = ''

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),      # Randomly rotate by up to 180 degrees
    #transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur
    #transforms.Lambda(threshold_image),  # Apply the custom thresholding function
    #transforms.Lambda(hysteresis_thresholding),  # Apply the edge detection
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust color
    transforms.Resize((224, 224)),
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize to 224x224
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize to 224x224
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

def plot_class_distribution(dataset, title):
    class_counts = Counter(dataset.targets)
    class_names = dataset.classes
    counts = [class_counts[i] for i in range(len(class_names))]

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts)
    plt.xlabel(title + ' Dataset for Mango Classifications')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Mango Classification')
    plt.xticks(rotation=45)

    # Save as PNG instead of only showing
    save_path = os.path.join(data_dir, f"{title.lower()}_class_distribution_bruises.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")
    
def create_dataloaders():
    train_data_path = os.path.join(data_dir, 'train/bruises')
    test_data_path = os.path.join(data_dir, 'test/bruises')
    val_data_path = os.path.join(data_dir, 'val/bruises')

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transforms)
    val_dataset = datasets.ImageFolder(root=val_data_path, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    plot_class_distribution(train_dataset, "Train")
    plot_class_distribution(test_dataset, "Test")
    plot_class_distribution(val_dataset, "Valid")
    return train_loader, test_loader, train_dataset.classes, val_loader

def evaluate_model(model, test_loader, class_names):
    model.eval()
    y_true = []
    y_pred = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1 Score: {f1:.4f}")

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    save_path = os.path.join(data_dir, "confusion_matrix_bruises.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")
    
def train_model(num_epochs):
    train_loader, test_loader, class_names, val_loader = create_dataloaders()
    
    # CHANGEME to correct efficientnet model
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track validation metrics
    val_loss_list = []
    val_acc_list = []
    epoch_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)

        epoch_list.append(epoch + 1)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(val_accuracy)

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, val_loss_list, label="Validation Loss", marker='o')
    plt.plot(epoch_list, val_acc_list, label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation Loss & Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(data_dir, "val_loss_accuracy_curve_bruises.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

    file_path = os.path.join(data_dir, "bruises_b0.pth")
    torch.save(model.state_dict(), file_path)
    print("Model saved successfully.")

    evaluate_model(model, test_loader, class_names)
    
def main():
    # CHANGE ME
    EPOCHS = 1
    log_path = os.path.join(data_dir, "log_bruises.txt")
    with open(log_path, "w") as f:
        sys.stdout = f
        train_model(EPOCHS)
        
    sys.stdout = sys.__stdout__
    print(f"Training log saved to: {log_path}")

if __name__ == "__main__":
    main()