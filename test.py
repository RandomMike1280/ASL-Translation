import numpy as np

dataset = np.load(r'dataset/more_datasets/1.npz')
distances = dataset['distances']
labels = dataset['frames']

for v in labels:
    print(v)