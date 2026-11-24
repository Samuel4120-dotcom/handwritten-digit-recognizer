import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. Look at the first 10 images with their labels
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# 2. Check how many examples of each digit there are
digits, counts = np.unique(y_train, return_counts=True)
for d, c in zip(digits, counts):
    print(f"Digit {d}: {c} images")

# 3. Check the pixel value range
print("Min pixel value:", x_train.min())
print("Max pixel value:", x_train.max())
print("Data type:", x_train.dtype)