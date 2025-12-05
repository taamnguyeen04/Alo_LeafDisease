# 🌿 Hướng dẫn sử dụng Hệ thống Two-Stage Classification

## 📋 Tổng quan

Hệ thống phân loại bệnh lá cây sử dụng phương pháp **Two-Stage Classification** để ưu tiên nhận diện đúng loại cây trước, sau đó mới phân loại bệnh cụ thể.

### 🎯 Ưu điểm của phương pháp này:
1. **Độ chính xác cao hơn**: Nhận diện đúng loại cây trước giúp tránh nhầm lẫn giữa các loại cây
2. **Chuyên biệt hóa**: Mỗi loại cây có model riêng để phân loại bệnh → độ chính xác cao hơn
3. **Dễ mở rộng**: Có thể thêm loại cây mới mà không ảnh hưởng các model cũ
4. **Giải thích được**: Người dùng biết rõ hệ thống đã nhận diện loại cây với độ tin cậy bao nhiêu

---

## 🚀 Hướng dẫn Training

### Bước 1: Chuẩn bị dữ liệu cho Stage 1 (Plant Type Classification)

```bash
python prepare_stage1_data.py
```

**🎯 Script tự động nhận diện cấu trúc dữ liệu:**

**Trường hợp 1:** Dữ liệu đã có split train/test sẵn
```
PlantVillage_Split/
├── train/
│   ├── apple_healthy/
│   ├── tomato_early_blight/
│   └── ...
└── test/
    └── ...
```
→ Script sẽ nhóm các classes theo plant type

**Trường hợp 2:** Dữ liệu chưa split (chỉ có folders class)
```
PlantVillage_Split/
├── apple_healthy/
├── tomato_early_blight/
└── ...
```
→ Script sẽ **tự động split** thành train/test (80/20) và nhóm theo plant type

**Các tính năng:**
- ✅ Nhóm các class theo loại cây (8 loại: apple, corn, grape, mango, peach, pepper, potato, tomato)
- ✅ **Tự động split** train/test nếu chưa có
- ✅ **Data Augmentation** để tăng số lượng ảnh training (mặc định: 3 ảnh augmented/ảnh gốc)
- ✅ Tạo dataset mới tại `C:\Users\tam\Documents\data\PlantVillage_Stage1_PlantType`

**Cấu hình:**

Trong file `prepare_stage1_data.py`, bạn có thể điều chỉnh:

```python
ENABLE_AUGMENTATION = True  # Bật/tắt augmentation
NUM_AUGMENTS = 3            # Số ảnh augmented cho mỗi ảnh gốc
TRAIN_RATIO = 0.8           # Tỷ lệ train/test (chỉ dùng khi auto split)
```

**Các kỹ thuật augmentation được áp dụng:**
- ✅ Random Horizontal Flip + Rotation
- ✅ Random Vertical Flip + Rotation
- ✅ Color Jitter (brightness, contrast, saturation)
- ✅ Random Rotation (các góc khác nhau)
- ✅ Kết hợp Flip + Color adjustment

**Output:**
```
PlantVillage_Stage1_PlantType/
├── train/
│   ├── apple/
│   ├── corn/
│   ├── grape/
│   ├── mango/
│   ├── peach/
│   ├── pepper/
│   ├── potato/
│   └── tomato/
├── test/
│   └── (tương tự)
└── train_balance/
    └── (tương tự)
```

### Bước 2: Train Stage 1 Model (Plant Type Classifier)

```bash
python train_stage1.py
```

**Cấu hình:**
- Batch size: 32
- Learning rate: 0.001
- Epochs: 20
- Model: ViT-B/16 (Vision Transformer)
- Số classes: 8 (plant types)

**Mục tiêu:** Độ chính xác > 95%

**Output:**
- Logs: `./logs_stage1/`
- Checkpoints: `./checkpoints_stage1/best.pt` và `./checkpoints_stage1/last.pt`

### Bước 3: Train Stage 2 Models (Disease Classifiers)

```bash
python train_stage2.py
```

Script này sẽ tự động train **8 models riêng biệt**, mỗi model cho một loại cây.

**Cấu hình:**
- Batch size: 32
- Learning rate: 0.001
- Epochs: 15 cho mỗi model
- Model: ViT-B/16

**Output:**
```
./checkpoints_stage2_apple/best.pt
./checkpoints_stage2_corn/best.pt
./checkpoints_stage2_grape/best.pt
./checkpoints_stage2_mango/best.pt
./checkpoints_stage2_peach/best.pt
./checkpoints_stage2_pepper/best.pt
./checkpoints_stage2_potato/best.pt
./checkpoints_stage2_tomato/best.pt
```

Và logs tương ứng:
```
./logs_stage2_apple/
./logs_stage2_corn/
...
```

### Bước 4: Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động load:
- Stage 1 model từ `./checkpoints_stage1/best.pt`
- Stage 2 models từ `./checkpoints_stage2_{plant_type}/best.pt`

---

## 📊 Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │   STAGE 1: Plant Classifier  │
          │   (8 classes: apple, corn,   │
          │    grape, mango, peach,      │
          │    pepper, potato, tomato)   │
          └──────────────┬───────────────┘
                         │
                         │ Predict: "tomato"
                         │ Confidence: 0.95
                         ▼
          ┌──────────────────────────────┐
          │ STAGE 2: Disease Classifier  │
          │   (Tomato-specific model)    │
          │   - tomato_bacterial_spot    │
          │   - tomato_early_blight      │
          │   - tomato_healthy           │
          │   - ...                      │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │      FINAL PREDICTION        │
          │  Plant: tomato (95%)         │
          │  Disease: early_blight (92%) │
          └──────────────────────────────┘
```

---

## 🔧 Cấu trúc thư mục

```
Alo_LeafDisease/
├── prepare_stage1_data.py      # Script chuẩn bị dữ liệu Stage 1
├── train_stage1.py              # Training Plant Type Classifier
├── train_stage2.py              # Training Disease Classifiers
├── app.py                       # Streamlit application
│
├── checkpoints_stage1/          # Stage 1 checkpoints
│   ├── best.pt
│   └── last.pt
│
├── checkpoints_stage2_apple/    # Stage 2 checkpoints cho apple
│   ├── best.pt
│   └── last.pt
│
├── checkpoints_stage2_corn/     # ... và các loại cây khác
├── checkpoints_stage2_grape/
├── checkpoints_stage2_mango/
├── checkpoints_stage2_peach/
├── checkpoints_stage2_pepper/
├── checkpoints_stage2_potato/
├── checkpoints_stage2_tomato/
│
├── logs_stage1/                 # TensorBoard logs cho Stage 1
└── logs_stage2_*/               # TensorBoard logs cho Stage 2
```

---

## 📈 Monitoring với TensorBoard

### Xem kết quả training Stage 1:
```bash
tensorboard --logdir=./logs_stage1
```

### Xem kết quả training Stage 2 (ví dụ cho tomato):
```bash
tensorboard --logdir=./logs_stage2_tomato
```

### Xem tất cả:
```bash
tensorboard --logdir=./ --logdir_spec stage1:./logs_stage1,apple:./logs_stage2_apple,tomato:./logs_stage2_tomato
```

---

## 🎯 Class Mapping

### Stage 1: Plant Types (8 classes)
```python
['apple', 'corn', 'grape', 'mango', 'peach', 'pepper', 'potato', 'tomato']
```

### Stage 2: Disease Classes (theo từng loại cây)

**Apple (4 classes):**
- apple_apple_scab
- apple_black_rot
- apple_cedar_apple_rust
- apple_healthy

**Corn (4 classes):**
- corn_maize_cercospora_leaf_spot_gray_leaf_spot
- corn_maize_common_rust
- corn_maize_healthy
- corn_maize_northern_leaf_blight

**Grape (4 classes):**
- grape_black_rot
- grape_esca_black_measles
- grape_healthy
- grape_leaf_blight_isariopsis_leaf_spot

**Mango (8 classes):**
- mango_anthracnose
- mango_bacterial_canker
- mango_cutting_weevil
- mango_die_back
- mango_gall_midge
- mango_healthy
- mango_powdery_mildew
- mango_sooty_mould

**Peach (2 classes):**
- peach_bacterial_spot
- peach_healthy

**Pepper (2 classes):**
- pepper_bell_bacterial_spot
- pepper_bell_healthy

**Potato (3 classes):**
- potato_early_blight
- potato_healthy
- potato_late_blight

**Tomato (10 classes):**
- tomato_bacterial_spot
- tomato_early_blight
- tomato_healthy
- tomato_late_blight
- tomato_leaf_mold
- tomato_septoria_leaf_spot
- tomato_spider_mites_twospotted_spider_mite
- tomato_target_spot
- tomato_tomato_mosaic_virus
- tomato_tomato_yellow_leaf_curl_virus

---

## 💡 Tips & Best Practices

### 1. Data Augmentation
✅ **Đã tích hợp sẵn trong `prepare_stage1_data.py`!**

Script tự động augment data với các kỹ thuật:
- Random flips (horizontal/vertical)
- Random rotations (10-25 độ)
- Color jittering (brightness, contrast, saturation)

**Điều chỉnh mức độ augmentation:**

```python
# Trong prepare_stage1_data.py
ENABLE_AUGMENTATION = True
NUM_AUGMENTS = 5  # Tăng lên 5 để có nhiều data hơn

# Hoặc tắt augmentation nếu data đã đủ
ENABLE_AUGMENTATION = False
```

**Khi nào nên dùng augmentation:**
- ✅ Dataset nhỏ (< 500 ảnh/class)
- ✅ Data imbalance giữa các classes
- ✅ Muốn model robust hơn với các biến thể của ảnh

**Khi nào KHÔNG nên dùng:**
- ❌ Dataset đã rất lớn (> 5000 ảnh/class)
- ❌ Storage bị giới hạn
- ❌ Training time quá lâu

### 2. Learning Rate Scheduling
Đã tích hợp `ReduceLROnPlateau` để tự động giảm learning rate khi accuracy không cải thiện.

### 3. Early Stopping
Có thể thêm early stopping để tránh overfitting:
```python
patience = 5
no_improve_count = 0

if accuracy > best_acc:
    best_acc = accuracy
    no_improve_count = 0
else:
    no_improve_count += 1
    if no_improve_count >= patience:
        print("Early stopping!")
        break
```

### 4. Confidence Threshold
Trong production, nên set threshold cho confidence:
- Stage 1: Nếu confidence < 0.7 → yêu cầu ảnh rõ hơn
- Stage 2: Nếu confidence < 0.6 → cảnh báo kết quả không chắc chắn

---

## ⚠️ Troubleshooting

### 1. Model không load được
- Kiểm tra đường dẫn checkpoint
- Kiểm tra số lượng classes khớp với model

### 2. Out of Memory (OOM)
- Giảm batch_size từ 32 xuống 16 hoặc 8
- Giảm num_workers từ 6 xuống 2 hoặc 0

### 3. Accuracy thấp cho Stage 1
- Train thêm epochs (tăng từ 20 lên 30-40)
- Kiểm tra data có balanced không
- Thử learning rate khác (0.0001 hoặc 0.0005)

### 4. Accuracy thấp cho Stage 2 (một loại cây cụ thể)
- Kiểm tra số lượng ảnh training cho loại cây đó
- Thử train riêng với epochs cao hơn
- Kiểm tra chất lượng ảnh

---

## 📝 Changelog

**Version 2.0 - Two-Stage Classification**
- ✅ Thêm Stage 1: Plant Type Classifier
- ✅ Thêm Stage 2: 8 Disease Classifiers riêng biệt
- ✅ Cập nhật app.py để support 2-stage inference
- ✅ Thêm visualization cho cả 2 giai đoạn
- ✅ Cải thiện độ chính xác nhận diện

**Version 1.0 - Single Model**
- Sử dụng 1 model duy nhất cho 37 classes

---

## 📧 Support

Nếu có vấn đề, liên hệ:
- GitHub: [Your GitHub]
- Email: [Your Email]

---

**Happy Training! 🚀**
