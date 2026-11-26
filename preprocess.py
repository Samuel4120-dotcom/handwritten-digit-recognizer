"""
preprocess.py - Image cleaner for the Handwritten Digit Recognizer (Pair 1)

Turns a messy mouse drawing into a clean image that matches MNIST:
  - grayscale
  - white ink on a solid black background
  - 28 x 28 pixels
  - pixel values are decimals between 0 and 1

Two output shapes:
  - mode="2d"   -> (28, 28)  square grid  (for the CNN / deep learning model)
  - mode="flat" -> (784,)    one flat row (for the classical ML model)
"""

import numpy as np
from PIL import Image

SIZE = 28        # final image is 28 x 28
DIGIT_BOX = 20   # MNIST digits sit inside a 20 x 20 box, centred in the 28 x 28 image


def _to_gray_array(image):
    """Accept a file path, PIL image, or numpy array. Return a 2D float array (0-255)."""
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGBA"))

    raw = np.asarray(image)
    arr = raw.astype("float64")

    # Decimal images (0-1) are scaled up to 0-255 so everything is handled the same way
    if np.issubdtype(raw.dtype, np.floating) and arr.size and arr.max() <= 1.0:
        arr = arr * 255.0

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
        if arr.shape[2] == 4:                      # RGBA (the Streamlit canvas gives this)
            rgb = rgb * (arr[:, :, 3:4] / 255.0)   # blend transparency onto black
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _make_white_on_black(gray):
    """Make sure the ink is white and the background is solid black. Returns values 0-1."""
    if np.median(gray) > 127:        # bright background -> dark ink on light paper -> flip it
        gray = 255.0 - gray
    gray = gray - gray.min()
    peak = gray.max()
    if peak > 0:
        gray = gray / peak           # brightest ink becomes exactly 1.0
    gray[gray < 0.1] = 0.0           # wipe faint noise so the background is pure black
    return gray


def _centre_digit(gray):
    """Crop to the digit, shrink it into a 20x20 box, and centre it in a 28x28 image."""
    rows = np.where(gray.max(axis=1) > 0)[0]
    cols = np.where(gray.max(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((SIZE, SIZE))                      # blank drawing

    digit = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = digit.shape
    scale = DIGIT_BOX / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    small = Image.fromarray((digit * 255).astype("uint8")).resize((new_w, new_h), Image.LANCZOS)
    small = np.asarray(small, dtype="float64") / 255.0

    # Place it so the digit's centre of mass sits in the middle, like MNIST
    ys, xs = np.indices(small.shape)
    total = small.sum()
    com_y = (ys * small).sum() / total
    com_x = (xs * small).sum() / total
    top = int(np.clip(round(13.5 - com_y), 0, SIZE - new_h))
    left = int(np.clip(round(13.5 - com_x), 0, SIZE - new_w))

    out = np.zeros((SIZE, SIZE))
    out[top:top + new_h, left:left + new_w] = small
    return out


def preprocess(image, mode="2d"):
    """
    Clean a drawing for the models.

    image: file path, PIL image, or numpy array (e.g. canvas_result.image_data)
    mode : "2d"   -> (28, 28) array for the CNN
           "flat" -> (784,) array for the classical model
    """
    gray = _to_gray_array(image)
    gray = _make_white_on_black(gray)
    clean = _centre_digit(gray)
    clean = np.clip(clean, 0.0, 1.0).astype("float32")

    if mode in ("2d", "cnn"):
        return clean
    if mode in ("flat", "classical"):
        return clean.reshape(784)
    raise ValueError('mode must be "2d" or "flat"')


def preprocess_for_cnn(image):
    return preprocess(image, "2d")


def preprocess_for_classical(image):
    return preprocess(image, "flat")


def is_blank(image):
    """True if nothing was drawn (the app can warn the user instead of predicting)."""
    return preprocess(image).max() == 0


# Proof of done: run  python preprocess.py   (or  python preprocess.py some_drawing.png)
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from PIL import ImageDraw

    if len(sys.argv) > 1:
        sample = Image.open(sys.argv[1])
    else:
        # Fake mouse drawing: a thick white "7" on a 280x280 black canvas, like the app
        sample = Image.new("RGB", (280, 280), "black")
        ImageDraw.Draw(sample).line([(70, 70), (200, 70), (120, 220)],
                                    fill="white", width=20, joint="curve")

    two_d = preprocess(sample, "2d")
    flat = preprocess(sample, "flat")

    print("Original size       :", sample.size)
    print("2D output shape     :", two_d.shape)
    print("Flat output shape   :", flat.shape)
    print("Pixel range         : min", two_d.min(), "/ max", two_d.max())
    print("Background (corner) :", two_d[0, 0], "(0.0 = black)")
    print("Brightest ink       :", two_d.max(), "(1.0 = white)")
    print("White-on-black      :", bool(two_d[0, 0] == 0 and two_d.max() == 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(sample)
    axes[0].set_title(f"Original {sample.size[0]}x{sample.size[1]}")
    axes[1].imshow(two_d, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Cleaned 28x28")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.show()