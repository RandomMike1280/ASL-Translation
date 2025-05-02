import os
import numpy as np
import random

# Get all .npz files from the dataset directory
dataset_dir = "./dataset/more_datasets"
npz_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]

if not npz_files:
    print("No .npz files found in the dataset directory")
else:
    # Randomly select one .npz file
    random_file = random.choice(npz_files)
    data = np.load(os.path.join(dataset_dir, random_file))
    
    # Get a random example from the dataset
    keys = list(data.keys())
    num_examples = len(data[keys[0]])
    random_idx = random.randint(0, num_examples - 1)
    
    print(f"Selected file: {random_file}")
    print("\nShapes:")
    for key in keys:
        print(f"{key}: {data[key][random_idx].shape}") # 'distances': (120,) \n 'label': 0
    
    print("\nTargets:")
    for key in keys:
        if 'target' in key.lower() or 'label' in key.lower():
            print(f"{key}: {data[key][random_idx]}") # 'label': 0
