import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # White digit on black background, like MNIST
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return img.reshape(28, 28, 1)

folder = "data/my_digits"
files = sorted(f for f in os.listdir(folder)
               if f.lower().endswith((".png", ".jpg", ".jpeg")))

images, labels = [], []
plt.figure(figsize=(10, 3))
for i, f in enumerate(files):
    img = preprocess_image(os.path.join(folder, f))
    images.append(img)
    labels.append(int(f[0]))
    plt.subplot(1, len(files), i + 1)
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.title(f"Label {labels[-1]}")
    plt.axis("off")

np.savez_compressed("data/processed/my_digits.npz",
                    x=np.array(images), y=np.array(labels))
print("Saved", len(images), "custom images")
plt.show()