import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.datasets import ImageFolder
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train():
    batch_size = 32
    lr = 0.001
    epochs = 10
    log_path = "./logs"
    checkpoint_path = "./checkpoints"
    num_classes = len(os.listdir(r"C:\Users\tam\Documents\data\PlantVillage_Split\test"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ViT_B_16_Weights.IMAGENET1K_V1  # hoặc IMAGENET1K_SWAG_E2E_V1 nếu muốn dùng input 384x384

    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model.to(device)

    # Transform
    transform = weights.transforms()

    # Dataset + DataLoader
    train_dataset = ImageFolder(root=r"C:\Users\tam\Documents\data\PlantVillage_Split\train_balance", transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=6,
        shuffle=True,
        drop_last=False
    )
    test_dataset = ImageFolder(root=r"C:\Users\tam\Documents\data\PlantVillage_Split\test", transform=transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=6,
        shuffle=False,
        drop_last=False
    )

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Log + checkpoint
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(log_path)
    best_acc = -100

    categories = os.listdir(r"C:\Users\tam\Documents\data\PlantVillage_Split\test")

    # ==== TRAINING LOOP ====
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="BLUE")
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, epochs, loss))
            writer.add_scalar("Train/loss", loss, epoch * len(train_dataloader) + i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # VALIDATION
        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                predictions = torch.argmax(output, dim=1)
                loss = criterion(output, labels)
                all_losses.append(loss.item())
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
            loss = np.mean(all_losses)

            # 🔥 Tính F1 thay cho Accuracy
            f1 = f1_score(all_labels, all_predictions, average="macro")
            # average="macro" = trung bình đều cho nhiều class, phù hợp dữ liệu unbalance
            conf_matrix = confusion_matrix(all_labels, all_predictions)

            print("Epoch {}/{}. Loss {:0.4f}. F1 {:0.4f}".format(epoch + 1, epochs, loss, f1))
            writer.add_scalar("Test/loss", loss, epoch)
            writer.add_scalar("Test/F1", f1, epoch)
            plot_confusion_matrix(writer, conf_matrix, categories, epoch)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, "last.pt"))
            if f1 > best_acc:   # đổi best_acc thành best_f1
                torch.save(checkpoint, os.path.join(checkpoint_path, "best.pt"))
                best_acc = f1


if __name__ == '__main__':
    train()
