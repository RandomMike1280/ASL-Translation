import numpy as np

data = np.load('dataset/distances/1.npz')

for key in data.files:
    print(f"{key}:\n{data[key]}\n")