# Data and Image Preparation

**Python version:** 3.12
**Setup:** python -m venv venv, then venv\Scripts\activate, then python -m pip install -r requirements.txt

## Data source
MNIST (70,000 handwritten digits, 28x28 grayscale), loaded via Keras.

## Preprocessing
1. Pixel values normalized from 0-255 to 0-1
2. Reshaped to (28, 28, 1)
3. Labels one-hot encoded (10 classes)
4. Split: 54,000 train / 6,000 validation / 10,000 test (random_state=42, stratified)

## Regenerate the data
python src/data_prep.py   (creates data/processed/digits.npz)

## Load the data
data = np.load("data/processed/digits.npz")
x_train, y_train = data["x_train"], data["y_train"]

## Augmentation (training set only)
Rotation 10 degrees, zoom 0.1, shifts 0.1 (see src/augment.py).
Never apply augmentation to validation or test data.