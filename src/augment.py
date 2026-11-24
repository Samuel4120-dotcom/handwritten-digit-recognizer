import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data = np.load("data/processed/digits.npz")
x_train = data["x_train"]

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

# Show one digit and 9 augmented versions of it
sample = x_train[0:1]
plt.figure(figsize=(10, 4))
plt.subplot(2, 5, 1)
plt.imshow(sample[0].reshape(28, 28), cmap="gray")
plt.title("Original")
plt.axis("off")

for i, batch in enumerate(datagen.flow(sample, batch_size=1)):
    plt.subplot(2, 5, i + 2)
    plt.imshow(batch[0].reshape(28, 28), cmap="gray")
    plt.title(f"Aug {i + 1}")
    plt.axis("off")
    if i == 8:
        break

plt.tight_layout()
plt.savefig("data/augmentation_preview.png")
plt.show()