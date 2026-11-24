import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Load the raw data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize: pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Reshape: add a channel dimension -> (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 4. Split off a validation set (10% of training data)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

# 5. One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

# 6. Print the final shapes
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_val:  ", x_val.shape, "y_val:  ", y_val.shape)
print("x_test: ", x_test.shape, "y_test: ", y_test.shape)
print("Pixel range:", x_train.min(), "to", x_train.max())

# 7. Save everything in one file
os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/digits.npz",
    x_train=x_train, y_train=y_train,
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=y_test,
)
print("Saved to data/processed/digits.npz")