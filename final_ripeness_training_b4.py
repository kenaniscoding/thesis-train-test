import os, sys, copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print("-----")
print(torch.version.cuda)
print(torch.__version__)

data_dir = ''
# ---- EARLY STOPPING ----
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
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

# ---- TRANSFORMS (EfficientNet-B4 expects 380x380) ----
IMG_SIZE = 380
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

# ---- PLOT CLASS DISTRIBUTION ----
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
    save_path = os.path.join(data_dir,"final", f"{title.lower()}_class_distribution_ripeness.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# ---- DATALOADERS ----
def create_dataloaders():
    train_data_path = os.path.join(data_dir, 'train/ripeness')
    test_data_path  = os.path.join(data_dir, 'test/ripeness')
    val_data_path   = os.path.join(data_dir, 'val/ripeness')

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transforms)
    test_dataset  = datasets.ImageFolder(root=test_data_path,  transform=test_transforms)
    val_dataset   = datasets.ImageFolder(root=val_data_path,   transform=val_transforms)

    # Batch size 16 is safer for 380x380 on common GPUs
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=0)

    plot_class_distribution(train_dataset, "Train")
    plot_class_distribution(test_dataset,  "Test")
    plot_class_distribution(val_dataset,   "Valid")
    return train_loader, test_loader, train_dataset.classes, val_loader

# ---- EVALUATION ----
def evaluate_model(model, test_loader, class_names):
    model.eval()
    y_true, y_pred, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall    = recall_score(y_true, y_pred, average="weighted")
    f1        = f1_score(y_true, y_pred, average="weighted")
    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    save_path = os.path.join(data_dir,"final", "confusion_matrix_ripeness.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# ---- TRAINING ----
def train_model(num_epochs, patience=5):
    train_loader, test_loader, class_names, val_loader = create_dataloaders()
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ✅ AMP GradScaler (new API)
    scaler = torch.amp.GradScaler("cuda")

    # ---- Scheduler: warmup (10% steps) + cosine decay (per-step) ----
    total_steps  = num_epochs * len(train_loader)
    warmup_steps = max(1, int(0.10 * total_steps))  # 10% warmup

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # linear warmup from ~0 to 1
            return float(current_step) / float(warmup_steps)
        # cosine decay from 1 -> 0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_loss_list, val_acc_list, epoch_list = [], [], []
    early_stopper = EarlyStopping(patience=patience, min_delta=0.001)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # per-step update

            running_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix(loss=running_loss/len(train_loader),
                                     lr=optimizer.param_groups[0]['lr'])

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {running_loss/len(train_loader):.4f}")

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        epoch_list.append(epoch + 1)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(val_accuracy)

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # ---- Early stopping check ----
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Restoring best model weights.")
            model.load_state_dict(early_stopper.best_model_wts)
            break

    # ---- Save learning curve ----
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, val_loss_list, label="Validation Loss", marker='o')
    plt.plot(epoch_list, val_acc_list,  label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation Loss & Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(data_dir,"final", "val_loss_accuracy_curve_ripeness.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

    # ---- Save best model ----
    file_path = os.path.join(data_dir,"final", "ripeness_b4.pth")
    torch.save(model.state_dict(), file_path)
    print("Best model saved successfully.")

    evaluate_model(model, test_loader, class_names)

# ---- MAIN ----
def main():
    log_path = os.path.join(data_dir,"final", "log_ripeness.txt")
    with open(log_path, "w") as f:
        sys.stdout = f
        train_model(num_epochs=30, patience=5)
    sys.stdout = sys.__stdout__
    print(f"Training log saved to: {log_path}")

if __name__ == "__main__":
    main()
