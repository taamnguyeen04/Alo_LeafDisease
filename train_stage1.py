"""
Stage 1: Training Plant Type Classifier
Mục tiêu: Phân loại 8 loại cây với độ chính xác cao (>95%)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.datasets import ImageFolder
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import shutil
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix - Plant Type")
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


def validate_model(model, test_dataloader, criterion, device):
    """Perform validation and return metrics"""
    model.eval()
    all_losses = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            predictions = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            
            all_losses.append(loss.item())
            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())
    
    val_loss = np.mean(all_losses)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="macro")
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return val_loss, accuracy, f1, conf_matrix


def train():
    # ===== CONFIGURATION =====
    batch_size = 32
    lr = 0.001
    epochs = 50
    eval_steps = 500  # Đánh giá sau mỗi 500 steps
    early_stopping_patience = 10  # Dừng sau 10 lần eval không improve
    
    log_path = "./logs_stage1"
    checkpoint_path = "./checkpoints_stage1"
    data_root = r"C:/Users/tam/Desktop/Data/leaf/PlantVillage_Stage1_PlantType"
    
    # 8 plant types
    num_classes = 8

    # ===== DEVICE & MODEL =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model.to(device)

    # ===== DATA =====
    transform = weights.transforms()

    # Check data paths
    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Train data not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Test data not found at: {test_path}")
    
    train_dataset = ImageFolder(root=train_path, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_dataset = ImageFolder(root=test_path, transform=transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Test samples: {len(test_dataset)}")
    print(f"📊 Steps per epoch: {len(train_dataloader)}")
    print(f"📊 Eval every {eval_steps} steps")

    # ===== LOSS & OPTIMIZER =====
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # ===== LOGGING & CHECKPOINT =====
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(log_path)
    
    # Early stopping
    best_acc = 0.0
    best_f1 = 0.0
    evals_no_improve = 0
    global_step = 0

    categories = sorted(os.listdir(test_path))
    print(f"🌿 Plant types: {categories}")
    print(f"🎯 Total classes: {num_classes}")
    print(f"⚙️  Batch size: {batch_size}")
    print(f"📚 Learning rate: {lr}")
    print(f"🛑 Early stopping patience: {early_stopping_patience} evals")
    print("\n" + "="*60)

    # ===== TRAINING LOOP =====
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        
        # === TRAINING PHASE ===
        model.train()
        train_losses = []
        progress_bar = tqdm(train_dataloader, colour="BLUE", desc=f"Training Epoch {epoch+1}")
        
        for step, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            global_step += 1
            
            # Log training loss
            writer.add_scalar("Train/loss_step", loss.item(), global_step)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'step': global_step,
                'best_acc': f'{best_acc:.4f}'
            })
            
            # === EVALUATION EVERY eval_steps ===
            if global_step % eval_steps == 0:
                print(f"\n🔍 Evaluating at step {global_step}...")
                
                val_loss, accuracy, f1, conf_matrix = validate_model(
                    model, test_dataloader, criterion, device
                )
                
                # Average train loss for this eval period
                avg_train_loss = np.mean(train_losses[-eval_steps:] if len(train_losses) >= eval_steps else train_losses)
                
                # === LOGGING ===
                print(f"📊 Step {global_step} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) | F1: {f1:.4f}")
                
                writer.add_scalar("Train/loss_avg", avg_train_loss, global_step)
                writer.add_scalar("Val/loss", val_loss, global_step)
                writer.add_scalar("Val/Accuracy", accuracy, global_step)
                writer.add_scalar("Val/F1", f1, global_step)
                
                # Plot confusion matrix every 5 evals
                if (global_step // eval_steps) % 5 == 0:
                    plot_confusion_matrix(writer, conf_matrix, categories, global_step)
                
                # === LEARNING RATE SCHEDULING ===
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Train/learning_rate", current_lr, global_step)
                scheduler.step(accuracy)
                
                # === CHECKPOINT SAVING ===
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                    "f1": f1,
                    "best_acc": best_acc
                }
                
                # Save last checkpoint
                torch.save(checkpoint, os.path.join(checkpoint_path, "last.pt"))
                
                # Save best checkpoint
                if accuracy > best_acc:
                    torch.save(checkpoint, os.path.join(checkpoint_path, "best.pt"))
                    print(f"✅ New best accuracy: {accuracy:.4f} (previous: {best_acc:.4f})")
                    best_acc = accuracy
                    best_f1 = f1
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1
                    print(f"⏳ No improvement for {evals_no_improve} evaluation(s)")
                
                # === EARLY STOPPING ===
                if evals_no_improve >= early_stopping_patience:
                    print(f"\n🛑 Early stopping triggered at step {global_step}")
                    print(f"   No improvement for {early_stopping_patience} consecutive evaluations")
                    print("\n" + "="*60)
                    print("🎉 Training completed (early stopped)!")
                    print(f"📈 Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
                    print(f"📈 Best F1 Score: {best_f1:.4f}")
                    print(f"📁 Checkpoints saved to: {checkpoint_path}")
                    print(f"📊 TensorBoard logs: {log_path}")
                    print("="*60)
                    return
                
                # Back to training mode
                model.train()

    # ===== TRAINING SUMMARY =====
    print("\n" + "="*60)
    print("🎉 Training completed!")
    print(f"📈 Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"📈 Best F1 Score: {best_f1:.4f}")
    print(f"📁 Checkpoints saved to: {checkpoint_path}")
    print(f"📊 TensorBoard logs: {log_path}")
    print("="*60)


if __name__ == '__main__':
    train()
