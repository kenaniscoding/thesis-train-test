import torch, os, sys
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter
import copy  # <-- Added for EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- EarlyStopping Class ----------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
# ------------------------------------------------------

# CHANGEME IMG DATASET DIR
data_dir = ''

# Image size for EfficientNetV2-M
IMG_SIZE = 480  

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=os.cpu_count() // 2)
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=os.cpu_count() // 2)

    plot_class_distribution(train_dataset, "Train")
    plot_class_distribution(test_dataset, "Test")
    plot_class_distribution(val_dataset, "Valid")
    return train_loader, test_loader, train_dataset.classes, val_loader

def evaluate_model(model, test_loader, class_names):
    model.eval()
    y_true, y_pred, all_probs = [], [], []

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
    
    # Load EfficientNetV2-M with pretrained weights
    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    model = efficientnet_v2_m(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ---- Scheduler: CosineAnnealingWarmRestarts ----
    # T_0 = number of epochs before first restart
    # T_mult = factor to increase cycle length after each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,        # restart every 5 epochs initially
        T_mult=2,     # double the cycle length each time
        eta_min=1e-6  # minimum LR
    )

    # AMP GradScaler (updated to torch.amp API)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Early stopping instance
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # ---- Best model tracking variables ----
    best_val_loss = float('inf')
    best_epoch = -1
    best_wts = copy.deepcopy(model.state_dict())

    val_loss_list, val_acc_list, epoch_list = [], [], []
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # AUTOCAST & SCALER
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Step scheduler per batch for smooth cosine curve
            scheduler.step(epoch + batch_idx / len(train_loader))

            running_loss += loss.item()
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            
            correct_train += (predicted == labels).sum().item()
            global_step += 1
            progress_bar.set_postfix(loss=loss.item(),
                                     lr=optimizer.param_groups[0]['lr'])
        # Calculate training metrics
        train_accuracy = correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)   
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
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

        # ---- Best model tracking update ----
        if avg_val_loss < best_val_loss - 1e-9:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_wts = copy.deepcopy(model.state_dict())

        # ---- Early stopping check ----
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    # Save learning curves
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

    # ---- Load and save best model after training ----
    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), os.path.join(data_dir, "bruises_v2m.pth"))
    print(f"Saved best model from epoch {best_epoch} with val_loss={best_val_loss:.4f}")

    evaluate_model(model, test_loader, class_names)    
def main():
    EPOCHS = 100
    log_path = os.path.join(data_dir, "log_bruises.txt")
    with open(log_path, "w") as f:
        sys.stdout = f
        train_model(EPOCHS)
        
    sys.stdout = sys.__stdout__
    print(f"Training log saved to: {log_path}")

if __name__ == "__main__":
    main()