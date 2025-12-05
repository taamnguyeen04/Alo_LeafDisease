from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("model/stage1.h5", compile=False)

class_names = open("model/stage1.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open("C:/Users/tam/Downloads/z7223463884265_0cdb1a0b798b81a32066a58c753e5ca9.jpg").convert("RGB")

size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

image_array = np.asarray(image)

normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

data[0] = normalized_image_array

prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

if "apple" in class_name:
    model_2 = load_model("model/apple.h5", compile=False)
    class_names_2 = open("model/apple.txt", "r").readlines()

elif "mango" in class_name:
    model_2 = load_model("model/mango.h5", compile=False)
    class_names_2 = open("model/mango.txt", "r").readlines()

elif "tomato" in class_name:
    model_2 = load_model("model/tomato.h5", compile=False)
    class_names_2 = open("model/tomato.txt", "r").readlines()

prediction_2 = model_2.predict(data)
index_2 = np.argmax(prediction_2)
class_name_2 = class_names_2[index_2]
confidence_score_2 = prediction_2[0][index_2]  
print("Class:", class_name_2[2:], end="")
print("Confidence Score:", confidence_score_2)