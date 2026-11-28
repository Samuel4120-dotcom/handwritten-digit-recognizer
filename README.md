# Handwritten Digit Recognizer

Draw a digit (0-9) with your mouse and watch two AI models guess which number it is, live.

Built by ML Group 3 using the MNIST dataset of 70,000 handwritten digits.

## Features

- Interactive drawing canvas (Streamlit)
- Choose between two models:
  - **CNN Model (Deep Learning)**: our most accurate model (98%+ on test data)
  - **Baseline Model (Classical ML)**: the better of K-Nearest Neighbors and Random Forest
- Shows the predicted digit and a bar chart of confidence for all 10 digits

## Project Structure

```
handwritten-digit-recognizer/
├── app.py                 # Streamlit drawing app (Pair 4)
├── preprocess.py          # Cleans drawings into 28x28 images (Pair 1)
├── train_baseline.py      # Trains KNN and Random Forest (Pair 2)
├── train_cnn.py           # Trains the CNN (Pair 3)
├── evaluate.py            # Tests both models, makes charts (Pair 5)
├── models/
│   ├── baseline_model.pkl
│   └── cnn_model.keras
├── results/
│   ├── cnn_training_curve.png
│   └── confusion_matrix.png
└── README.md
```

## How to Run the App

1. **Clone the repository** (or pull the latest changes if you already have it):
   ```
   git clone https://github.com/Samuel4120-dotcom/handwritten-digit-recognizer.git
   cd handwritten-digit-recognizer
   ```

2. **Install the required packages** (Python 3.9 to 3.11 recommended):
   ```
   pip install streamlit streamlit-drawable-canvas tensorflow scikit-learn joblib numpy pillow matplotlib
   ```

3. **Start the app:**
   ```
   streamlit run app.py
   ```

4. Your browser will open the app. Draw a digit in the black box, pick a model from the dropdown, and click **Predict**.

> Tip: Draw the digit large and centered with a thick stroke for the best results.

## How to Evaluate the Models

After the trained models are in the `models/` folder, run:

```
python evaluate.py
```

This tests both models on the 10,000 MNIST test images, prints their accuracy, and saves confusion matrices in the `results/` folder.

## How It Works

1. **Draw**: the app captures a 280x280 drawing (white ink on black).
2. **Clean**: `preprocess.py` converts it to grayscale, shrinks it to 28x28, and scales pixel values between 0 and 1.
3. **Predict**: the cleaned image goes to the selected model. The CNN receives a 28x28 grid, and the baseline model receives a flat row of 784 numbers.
4. **Display**: the app shows the winning digit and a confidence bar chart.

## Results

| Model | Test Accuracy |
|-------|---------------|
| Baseline (KNN / Random Forest) | _fill in after evaluation_ |
| CNN | _fill in after evaluation_ |

See `results/confusion_matrix.png` for where each model gets confused.

## Team

ML Group 3: Pair 1 (Preprocessing), Pair 2 (Classical ML), Pair 3 (CNN), Pair 4 (Drawing App), Pair 5 (Testing, Report and Slides), with Gideon as GitHub repo maintainer.
