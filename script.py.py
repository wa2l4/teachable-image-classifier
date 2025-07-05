import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

uploaded = files.upload()
image_name = list(uploaded.keys())[0]

model = tensorflow.keras.models.load_model('keras_model.h5')

with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

image = Image.open(image_name).convert("RGB")
image = image.resize((224, 224), Image.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.expand_dims(normalized_image_array, axis=0)

prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence = prediction[0][index]

print(f"{class_name} ({confidence:.2f})")

plt.imshow(Image.open(image_name))
plt.title(f"{class_name} ({confidence:.2f})")
plt.axis('off')
plt.show()
