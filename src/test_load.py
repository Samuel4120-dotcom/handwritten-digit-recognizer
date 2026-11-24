import numpy as np

data = np.load("data/processed/digits.npz")
for name in data.files:
    print(name, data[name].shape)