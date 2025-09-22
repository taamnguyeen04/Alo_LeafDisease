import matplotlib.pyplot as plt

# Số lượng ảnh mỗi class (train + test)
class_counts = {
    "mango_anthracnose": 500,
    "mango_bacterial_canker": 500,
    "mango_cutting_weevil": 500,
    "mango_die_back": 500,
    "mango_gall_midge": 500,
    "mango_healthy": 500,
    "mango_powdery_mildew": 500,
    "mango_sooty_mould": 500,
    "pepper_bell_bacterial_spot": 997,
    "pepper_bell_healthy": 1478,
    "potato_early_blight": 1000,
    "potato_healthy": 152,
    "potato_late_blight": 1000,
    "tomato_bacterial_spot": 2127,
    "tomato_early_blight": 1000,
    "tomato_healthy": 1591,
    "tomato_late_blight": 1909,
    "tomato_leaf_mold": 952,
    "tomato_septoria_leaf_spot": 1771,
    "tomato_spider_mites_two_spotted_spider_mite": 1676,
    "tomato_target_spot": 1404,
    "tomato_tomato_mosaic_virus": 373,
    "tomato_tomato_yellowleaf_curl_virus": 3209
}

# Vẽ biểu đồ
plt.figure(figsize=(15,6))
plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
plt.xticks(rotation=90)
plt.ylabel("Số lượng ảnh")
plt.title("Số lượng ảnh mỗi lớp trong PlantVillage (bị imbalance)")
plt.tight_layout()
plt.show()