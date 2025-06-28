import numpy as np

# load npz file
data = np.load(r"C:\Users\angel\Desktop\Chị Huyền\dataset\distances\1.npz")

print(data['distances'][0].shape)