# Image Classification: Notebook vs. Pen (Teachable Machine + TensorFlow)

This project uses a machine learning model trained with Google's Teachable Machine to classify images as either a **notebook** or a **pen**.

## 📦 Files Included

- `keras_model.h5` – Trained Keras model exported from Teachable Machine.
- `labels.txt` – Class labels used by the model (`Notebook`, `Pen`).
- `predict_colab.py` – Python script for running predictions in Google Colab.
- Example test image(s)

## 🧠 Model Description

- **Classes**:  
  - 📓 `Notebook`  
  - 🖊️ `Pen`
- Model trained with custom images using Teachable Machine.
- Input image size: 224x224 pixels, RGB.
- Exported in Keras `.h5` format.

## 🚀 How to Use (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the following files:
   - `keras_model.h5`
   - `labels.txt`
3. Run the provided Python code.
4. Upload any image of a notebook or a pen.
5. The model will output the predicted class and display the image.

