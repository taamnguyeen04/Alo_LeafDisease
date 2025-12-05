"""
Stage 2: Training Disease Classifiers for each Plant Type
Mục tiêu: Sau khi biết loại cây, phân loại bệnh cụ thể
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

def plot_confusion_matrix(writer, cm, class_names, epoch, plant_type):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    figure = plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title(f"Confusion matrix - {plant_type}")
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


def get_disease_mapping():
    """
    Mapping từ plant type sang các disease classes
    """
    return {
        'apple': ['apple_apple_scab', 'apple_black_rot', 'apple_cedar_apple_rust', 'apple_healthy'],
        'corn': ['corn_maize_cercospora_leaf_spot_gray_leaf_spot', 'corn_maize_common_rust', 
                 'corn_maize_healthy', 'corn_maize_northern_leaf_blight'],
        'grape': ['grape_black_rot', 'grape_esca_black_measles', 'grape_healthy', 
                  'grape_leaf_blight_isariopsis_leaf_spot'],
        'mango': ['mango_anthracnose', 'mango_bacterial_canker', 'mango_cutting_weevil', 
                  'mango_die_back', 'mango_gall_midge', 'mango_healthy', 'mango_powdery_mildew', 
                  'mango_sooty_mould'],
        'peach': ['peach_bacterial_spot', 'peach_healthy'],
        'pepper': ['pepper_bell_bacterial_spot', 'pepper_bell_healthy'],
        'potato': ['potato_early_blight', 'potato_healthy', 'potato_late_blight'],
        'tomato': ['tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy', 
                   'tomato_late_blight', 'tomato_leaf_mold', 'tomato_septoria_leaf_spot', 
                   'tomato_spider_mites_twospotted_spider_mite', 'tomato_target_spot', 
                   'tomato_tomato_mosaic_virus', 'tomato_tomato_yellow_leaf_curl_virus']
    }


def train_plant_specific_model(plant_type, disease_classes, data_root, epochs=15):
    """
    Train một model riêng cho một loại cây cụ thể
    """
    print(f"\n{'='*80}")
    print(f"Training model for: {plant_type.upper()}")
    print(f"Number of disease classes: {len(disease_classes)}")
    print(f"Classes: {disease_classes}")
    print(f"{'='*80}\n")
    
    batch_size = 32
    lr = 0.001
    log_path = f"./logs_stage2_{plant_type}"
    checkpoint_path = f"./checkpoints_stage2_{plant_type}"
    
    num_classes = len(disease_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ViT_B_16_Weights.IMAGENET1K_V1

    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model.to(device)

    # Transform
    transform = weights.transforms()

    # Dataset + DataLoader
    # Filter để chỉ lấy classes thuộc plant_type này
    def filter_classes(root, allowed_classes):
        """Tạo dataset chỉ với các classes được phép"""
        from torch.utils.data import Dataset
        import os
        from PIL import Image
        
        samples = []
        class_to_idx = {cls: idx for idx, cls in enumerate(allowed_classes)}
        
        for class_name in os.listdir(root):
            if class_name not in allowed_classes:
                continue
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    samples.append((img_path, class_to_idx[class_name]))
        
        return samples, class_to_idx
    
    class FilteredImageFolder(torch.utils.data.Dataset):
        def __init__(self, root, allowed_classes, transform=None):
            self.samples, self.class_to_idx = filter_classes(root, allowed_classes)
            self.transform = transform
            self.classes = allowed_classes
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            from PIL import Image
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    train_dataset = FilteredImageFolder(
        root=os.path.join(data_root, "train_balance"),
        allowed_classes=disease_classes,
        transform=transform
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=6,
        shuffle=True,
        drop_last=False
    )
    
    test_dataset = FilteredImageFolder(
        root=os.path.join(data_root, "test"),
        allowed_classes=disease_classes,
        transform=transform
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=6,
        shuffle=False,
        drop_last=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Log + checkpoint
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(log_path)
    best_acc = -100

    # ==== TRAINING LOOP ====
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="GREEN")
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            progress_bar.set_description(f"[{plant_type}] Epoch {epoch + 1}/{epochs}. Loss {loss:.4f}")
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

            # Tính accuracy và F1
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average="macro")
            conf_matrix = confusion_matrix(all_labels, all_predictions)

            print(f"[{plant_type}] Epoch {epoch + 1}/{epochs}. Loss {loss:.4f}. Accuracy {accuracy:.4f}. F1 {f1:.4f}")
            writer.add_scalar("Test/loss", loss, epoch)
            writer.add_scalar("Test/Accuracy", accuracy, epoch)
            writer.add_scalar("Test/F1", f1, epoch)
            plot_confusion_matrix(writer, conf_matrix, disease_classes, epoch, plant_type)
            
            # Update learning rate
            scheduler.step(accuracy)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": accuracy,
                "f1": f1,
                "plant_type": plant_type,
                "disease_classes": disease_classes
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, "last.pt"))
            if accuracy > best_acc:
                torch.save(checkpoint, os.path.join(checkpoint_path, "best.pt"))
                best_acc = accuracy
                print(f"✅ [{plant_type}] New best accuracy: {best_acc:.4f}")

    print(f"\n🎉 [{plant_type}] Training completed! Best accuracy: {best_acc:.4f}")
    return best_acc


def train_all():
    """
    Train tất cả các model cho từng loại cây
    """
    data_root = r"C:\Users\tam\Documents\data\PlantVillage_Split"
    disease_mapping = get_disease_mapping()
    
    results = {}
    
    print("\n" + "="*80)
    print("STAGE 2: TRAINING DISEASE CLASSIFIERS")
    print("="*80)
    
    for plant_type, disease_classes in disease_mapping.items():
        best_acc = train_plant_specific_model(
            plant_type=plant_type,
            disease_classes=disease_classes,
            data_root=data_root,
            epochs=15
        )
        results[plant_type] = best_acc
    
    print("\n" + "="*80)
    print("SUMMARY - STAGE 2 RESULTS")
    print("="*80)
    for plant_type, acc in results.items():
        print(f"{plant_type:15s}: {acc:.4f} ({acc*100:.2f}%)")
    print("="*80)


if __name__ == '__main__':
    train_all()
