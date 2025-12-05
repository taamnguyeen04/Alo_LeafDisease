"""
Script để chuẩn bị dữ liệu cho Stage 1: Plant Type Classification
Tạo dataset với 8 classes chính: apple, corn, grape, mango, peach, pepper, potato, tomato
Có hỗ trợ Data Augmentation để tăng số lượng ảnh training
"""
import os
import shutil
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms
import random
import io

import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageFilter

def get_augmentation_transforms():

    def random_shadow(img):
        # Bóng mờ nhẹ
        w, h = img.size
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
        shadow = Image.new('RGB', img.size, (0, 0, 0))
        mask = Image.new('L', img.size, 0)
        Image.Image.paste(mask, Image.new('L', (x2-x1, y2-y1), random.randint(40, 80)), (x1, y1))
        return Image.composite(img, shadow, mask)

    class ShadowTransform:
        def __call__(self, img):
            return random_shadow(img)

    # JPEG corruption
    class JpegCompression:
        def __call__(self, img):
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=random.randint(30, 90))
            buf.seek(0)
            return Image.open(buf).convert("RGB")

    return [
        transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(0.7),
            transforms.ColorJitter(0.4, 0.4, 0.3, 0.2),
        ]),
        transforms.Compose([
            transforms.RandomVerticalFlip(0.7),
            transforms.RandomAffine(20, translate=(0.1,0.1)),
        ]),
        transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.GaussianBlur(kernel_size=5),
        ]),
        transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ]),
        transforms.Compose([
            ShadowTransform(),
        ]),
        transforms.Compose([
            JpegCompression(),
        ])
    ]


def augment_image(img_path, output_dir, base_name, num_augments=3):
    try:
        img = Image.open(img_path).convert('RGB')
        augment_transforms = get_augmentation_transforms()
        
        selected_transforms = random.sample(augment_transforms, min(num_augments, len(augment_transforms)))
        
        augmented_paths = []
        for i, transform in enumerate(selected_transforms):
            aug_img = transform(img)
            name, ext = os.path.splitext(base_name)
            aug_name = f"{name}_aug{i+1}{ext}"
            aug_path = os.path.join(output_dir, aug_name)
            aug_img.save(aug_path)
            augmented_paths.append(aug_path)
        
        return augmented_paths
    except Exception as e:
        print(f"Warning: Không thể augment {img_path}: {e}")
        return []

def prepare_plant_type_data(source_dir, output_dir, augment=False, num_augments=3):
    """
    Chuẩn bị dữ liệu cho plant type classification
    Mỗi ảnh từ class "plant_disease" sẽ được copy vào class "plant"
    
    Args:
        source_dir: thư mục chứa dữ liệu gốc
        output_dir: thư mục output
        augment: có augment data hay không
        num_augments: số lượng ảnh augmented cho mỗi ảnh gốc (chỉ áp dụng cho train)
    """
    
    # Mapping từ class gốc sang plant type
    plant_mapping = {
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
    
    # Tạo reverse mapping
    class_to_plant = {}
    for plant, classes in plant_mapping.items():
        for cls in classes:
            class_to_plant[cls] = plant
    
    for split in ['train', 'test', 'train_balance']:
        source_split_dir = os.path.join(source_dir, split)
        if not os.path.exists(source_split_dir):
            print(f"Warning: {source_split_dir} không tồn tại, bỏ qua...")
            continue
            
        output_split_dir = os.path.join(output_dir, split)
        
        # Xóa và tạo lại thư mục output
        if os.path.exists(output_split_dir):
            shutil.rmtree(output_split_dir)
        os.makedirs(output_split_dir, exist_ok=True)
        
        # Đếm số ảnh cho mỗi plant type
        stats = defaultdict(int)
        augmented_stats = defaultdict(int)
        
        # Chỉ augment cho train và train_balance
        should_augment = augment and ('train' in split)
        
        # Duyệt qua các class gốc
        for class_name in os.listdir(source_split_dir):
            class_path = os.path.join(source_split_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # Lấy plant type từ class name
            if class_name not in class_to_plant:
                print(f"Warning: {class_name} không có trong mapping, bỏ qua...")
                continue
            
            plant_type = class_to_plant[class_name]
            
            # Tạo thư mục cho plant type nếu chưa có
            plant_dir = os.path.join(output_split_dir, plant_type)
            os.makedirs(plant_dir, exist_ok=True)
            
            # Copy tất cả ảnh từ class gốc sang thư mục plant type
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    # Đổi tên file để tránh trùng: plant_originalclass_imgname
                    new_img_name = f"{plant_type}_{class_name}_{img_name}"
                    dst_path = os.path.join(plant_dir, new_img_name)
                    shutil.copy(img_path, dst_path)
                    stats[plant_type] += 1
                    
                    # Augment data nếu cần
                    if should_augment:
                        augmented_paths = augment_image(img_path, plant_dir, new_img_name, num_augments)
                        augmented_stats[plant_type] += len(augmented_paths)
        
        print(f"\n=== Split: {split} ===")
        for plant in sorted(stats.keys()):
            original = stats[plant]
            if should_augment:
                augmented = augmented_stats[plant]
                total = original + augmented
                print(f"{plant}: {original} original + {augmented} augmented = {total} images")
            else:
                print(f"{plant}: {original} images")
        
        if should_augment:
            print(f"Total: {sum(stats.values())} original + {sum(augmented_stats.values())} augmented = {sum(stats.values()) + sum(augmented_stats.values())} images")
        else:
            print(f"Total: {sum(stats.values())} images")

def auto_split_and_prepare(root_dir, output_dir, train_ratio=0.8, augment=False, num_augments=3, balance_data=False, max_samples_per_class=None):
    """
    Tự động split data và chuẩn bị cho Stage 1
    Dùng khi chưa có split train/test sẵn
    
    Args:
        balance_data: Có cân bằng số lượng ảnh giữa các plant types không
        max_samples_per_class: Số lượng ảnh tối đa cho mỗi plant type (trước khi augment)
    """
    import random
    from PIL import Image
    
    # Mapping từ class gốc sang plant type
    plant_mapping = {
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
    
    # Tạo reverse mapping
    class_to_plant = {}
    for plant, classes in plant_mapping.items():
        for cls in classes:
            class_to_plant[cls] = plant
    
    # Tạo thư mục output
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_stats = defaultdict(int)
    test_stats = defaultdict(int)
    train_aug_stats = defaultdict(int)
    
    # Collect all images per plant type first for balancing
    plant_images = defaultdict(list)  # plant_type -> [(class_name, img_path)]
    
    random.seed(42)
    
    # First pass: collect all images grouped by plant type
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        if class_name not in class_to_plant:
            print(f"Warning: {class_name} không có trong mapping, bỏ qua...")
            continue
        
        plant_type = class_to_plant[class_name]
        
        # Collect all image paths
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            plant_images[plant_type].append((class_name, img_path, img_name))
    
    # Calculate target number if balancing
    if balance_data or max_samples_per_class:
        if max_samples_per_class:
            target_num = max_samples_per_class
        else:
            # Use median to avoid being affected by outliers
            counts = [len(imgs) for imgs in plant_images.values()]
            target_num = int(np.median(counts))
        
        print(f"\n🔄 Balancing data: target {target_num} images per plant type (before augmentation)")
        
        # Balance by sampling
        for plant_type in plant_images:
            current_count = len(plant_images[plant_type])
            if current_count > target_num:
                print(f"   {plant_type}: {current_count} -> {target_num} (removing {current_count - target_num})")
                random.shuffle(plant_images[plant_type])
                plant_images[plant_type] = plant_images[plant_type][:target_num]
    
    # Second pass: process balanced images
    for plant_type, images_list in plant_images.items():
        # Tạo thư mục cho plant type
        train_plant_dir = os.path.join(train_dir, plant_type)
        test_plant_dir = os.path.join(test_dir, plant_type)
        os.makedirs(train_plant_dir, exist_ok=True)
        os.makedirs(test_plant_dir, exist_ok=True)
        
        # Shuffle all images for this plant type
        random.shuffle(images_list)
        
        # Split train/test
        split_idx = int(len(images_list) * train_ratio)
        if len(images_list) > 1 and split_idx == len(images_list):
            split_idx = len(images_list) - 1
        if split_idx == 0 and len(images_list) > 1:
            split_idx = 1
        
        train_images = images_list[:split_idx]
        test_images = images_list[split_idx:]
        
        # Copy train images
        for class_name, img_path, img_name in train_images:
            new_img_name = f"{plant_type}_{class_name}_{img_name}"
            dst_path = os.path.join(train_plant_dir, new_img_name)
            shutil.copy(img_path, dst_path)
            train_stats[plant_type] += 1
            
            # Augment train data
            if augment:
                augmented_paths = augment_image(img_path, train_plant_dir, new_img_name, num_augments)
                train_aug_stats[plant_type] += len(augmented_paths)
        
        # Copy test images (không augment)
        for class_name, img_path, img_name in test_images:
            new_img_name = f"{plant_type}_{class_name}_{img_name}"
            dst_path = os.path.join(test_plant_dir, new_img_name)
            shutil.copy(img_path, dst_path)
            test_stats[plant_type] += 1
    
    # In thống kê
    print("\n=== TRAIN SET ===")
    for plant in sorted(train_stats.keys()):
        original = train_stats[plant]
        if augment:
            augmented = train_aug_stats[plant]
            total = original + augmented
            print(f"{plant:10s}: {original:5d} original + {augmented:5d} augmented = {total:5d} images")
        else:
            print(f"{plant:10s}: {original:5d} images")
    
    if augment:
        total_train = sum(train_stats.values()) + sum(train_aug_stats.values())
        print(f"{'Total':10s}: {sum(train_stats.values()):5d} original + {sum(train_aug_stats.values()):5d} augmented = {total_train:5d} images")
    else:
        print(f"{'Total':10s}: {sum(train_stats.values()):5d} images")
    
    print("\n=== TEST SET ===")
    for plant in sorted(test_stats.keys()):
        print(f"{plant:10s}: {test_stats[plant]:5d} images")
    print(f"{'Total':10s}: {sum(test_stats.values()):5d} images")

if __name__ == "__main__":
    source_dir = r"C:/Users/tam/Desktop/Data/leaf/plantvillage dataset"
    output_dir = r"C:/Users/tam/Desktop/Data/leaf/PlantVillage_Stage1_PlantType"
    
    # ===== CẤU HÌNH =====
    ENABLE_AUGMENTATION = True   # Bật/tắt data augmentation
    NUM_AUGMENTS = 3             # Số ảnh augmented cho mỗi ảnh gốc
    TRAIN_RATIO = 0.8            # Tỷ lệ train/test
    
    # Data Balancing Options
    ENABLE_BALANCING = True      # Bật/tắt cân bằng dữ liệu
    MAX_SAMPLES_PER_CLASS = 3500 # Số ảnh tối đa cho mỗi plant type (None = auto balance theo median)
    
    print("🌿 Chuẩn bị dữ liệu cho Stage 1: Plant Type Classification")
    print(f"📁 Source: {source_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Train/Test ratio: {TRAIN_RATIO}/{1-TRAIN_RATIO}")
    print(f"🎨 Data Augmentation: {'ENABLED' if ENABLE_AUGMENTATION else 'DISABLED'}")
    if ENABLE_AUGMENTATION:
        print(f"   → Số ảnh augmented/ảnh gốc: {NUM_AUGMENTS}")
    print(f"⚖️  Data Balancing: {'ENABLED' if ENABLE_BALANCING else 'DISABLED'}")
    if ENABLE_BALANCING and MAX_SAMPLES_PER_CLASS:
        print(f"   → Max samples per class: {MAX_SAMPLES_PER_CLASS}")
    elif ENABLE_BALANCING:
        print(f"   → Auto balance (use median)")
    print("\n" + "="*60)
    
    # Kiểm tra cấu trúc thư mục
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if 'train' in subdirs and 'test' in subdirs:
        # Đã có split sẵn
        print("✓ Phát hiện đã có split train/test sẵn")
        prepare_plant_type_data(source_dir, output_dir, augment=ENABLE_AUGMENTATION, num_augments=NUM_AUGMENTS)
    else:
        # Chưa split, tự động split
        print("✓ Chưa có split train/test, sẽ tự động split")
        auto_split_and_prepare(
            source_dir, 
            output_dir, 
            train_ratio=TRAIN_RATIO, 
            augment=ENABLE_AUGMENTATION, 
            num_augments=NUM_AUGMENTS,
            balance_data=ENABLE_BALANCING,
            max_samples_per_class=MAX_SAMPLES_PER_CLASS
        )
    
    print("\n" + "="*60)
    print("✅ Hoàn thành!")
