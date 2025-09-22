import os
import random
import shutil
import cv2
import albumentations as A

# augmentation pipeline
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
])

def balance_dataset(src_root, dst_root, target_size=1000, seed=42, copy=True):
    """
    Tạo dataset cân bằng:
      - Nếu class > target_size  → undersample xuống target_size
      - Nếu class < target_size  → oversample (augmentation) lên target_size
      - Nếu class = target_size  → giữ nguyên
    """
    random.seed(seed)
    os.makedirs(dst_root, exist_ok=True)

    for cls in os.listdir(src_root):
        cls_path = os.path.join(src_root, cls)
        if not os.path.isdir(cls_path):
            continue

        files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        dst_cls = os.path.join(dst_root, cls)
        os.makedirs(dst_cls, exist_ok=True)

        n = len(files)

        # -------- case 1: undersample --------
        if n > target_size:
            chosen = random.sample(files, target_size)
            for f in chosen:
                src = os.path.join(cls_path, f)
                dst = os.path.join(dst_cls, f)
                if copy:
                    shutil.copy(src, dst)
                else:
                    shutil.move(src, dst)
            print(f"{cls}: {target_size} ảnh (undersampled từ {n})")

        # -------- case 2: oversample --------
        elif n < target_size:
            # copy toàn bộ ảnh gốc trước
            for f in files:
                shutil.copy(os.path.join(cls_path, f), os.path.join(dst_cls, f))

            count = n
            idx = 0
            while count < target_size:
                f = random.choice(files)
                img = cv2.imread(os.path.join(cls_path, f))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                aug = augment(image=img)["image"]

                aug_name = f"aug_{idx}_{f}"
                aug_path = os.path.join(dst_cls, aug_name)
                cv2.imwrite(aug_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
                idx += 1
                count += 1

            print(f"{cls}: {target_size} ảnh (oversampled từ {n})")

        # -------- case 3: giữ nguyên --------
        else:
            for f in files:
                shutil.copy(os.path.join(cls_path, f), os.path.join(dst_cls, f))
            print(f"{cls}: {n} ảnh (giữ nguyên)")

balance_dataset(r"C:\Users\tam\Documents\data\PlantVillage_Split\train", r"C:\Users\tam\Documents\data\PlantVillage_Split\train_balance", target_size=800)
