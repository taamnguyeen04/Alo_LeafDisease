import os, random, shutil

def split_dataset_move(root_dir, output_dir, train_ratio=0.8, seed=42, move_files=True):
    random.seed(seed)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        # ensure at least 1 test image if possible
        split_idx = int(len(images) * train_ratio)
        if len(images) > 1 and split_idx == len(images):
            split_idx = len(images) - 1
        if split_idx == 0 and len(images) > 1:
            split_idx = 1

        train_images = images[:split_idx]
        test_images = images[split_idx:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            if move_files:
                shutil.move(src, dst)
            else:
                shutil.copy(src, dst)
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            if move_files:
                shutil.move(src, dst)
            else:
                shutil.copy(src, dst)

        print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")


if __name__ == "__main__":
    root_dir = r"C:\Users\tam\Documents\data\PlantVillage"  # thư mục gốc dataset
    output_dir = r"C:\Users\tam\Documents\data\PlantVillage_Split"  # thư mục sau khi chia
    split_dataset_move(root_dir, output_dir, train_ratio=0.8)