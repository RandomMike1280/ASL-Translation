import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from translator import Encoder
from tokenizer import Tokenizer
from pre_vae import BottleneckT5Autoencoder

class ModelConfig:
    hidden_size = 256
    num_attention_heads = 4
    head_dim = 64
    kv_compress_dim = 128
    q_compress_dim = 256
    rope_dim = 32
    max_position_embeddings = 1024
    intermediate_size = 512
    num_layers = 2
    latent_dim = 1024
    input_dim = 1106

class NPZDataset(Dataset):
    def __init__(self, distances_dir, labels_dir, autoencoder, transform=None, chunk_size=256):
        self.samples = []
        self.transform = transform
        self.chunk_size = chunk_size
        self.autoencoder = autoencoder
        for fname in os.listdir(distances_dir):
            if not fname.endswith('.npz'):
                continue
            base = os.path.splitext(fname)[0]
            npz_path = os.path.join(distances_dir, fname)
            csv_path = os.path.join(labels_dir, base + '.csv')

            npz_file = np.load(npz_path)
            arrays = npz_file[npz_file.files[0]]
            # load labels per index
            if os.path.exists(csv_path):
                # load labels using pandas
                df = pd.read_csv(csv_path)
                # assume first col is index, second is label
                label_map = dict(zip(
                    df.iloc[:,0].astype(int).tolist(),
                    df.iloc[:,1].astype(str).tolist()
                ))
            else:
                label_map = {}
            # build token list and encoded label ids
            labels_list = [label_map.get(i, '<None>') for i in range(len(arrays))]

            # tensorize sequence
            seq_tensor = torch.tensor(arrays, dtype=torch.float)
            if self.transform:
                seq_tensor = self.transform(seq_tensor)
            self.samples.append((seq_tensor, labels_list))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, labels = self.samples[idx]
        
        # Randomly sample a chunk of size chunk_size
        start_idx = torch.randint(0, max(1, len(seq) - self.chunk_size), (1,)).item()
        end_idx = start_idx + self.chunk_size
        
        input_chunk = seq[start_idx:end_idx]
        label_chunk = labels[start_idx:end_idx]
        
        # Filter out <None> tokens from the labels to form the target sequence
        target_tokens = ' '.join([token for token in label_chunk if token != '<None>'])
        target_seq = self.autoencoder.embed(target_tokens)
        
        # The input to the model is the chunk of frames
        input_seq = input_chunk
        
        # generate time indices for each input timestep
        time_seq = torch.arange(input_seq.size(0), dtype=torch.long)
        
        return input_seq, target_seq, time_seq

def collate_fn(batch):
    # Pad sequences to the max length in the batch
    input_seqs, target_seqs, time_seqs = zip(*batch)
    
    # Pad input and time sequences
    padded_input_seqs = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True)
    padded_time_seqs = nn.utils.rnn.pad_sequence(time_seqs, batch_first=True)
    
    # Pad target sequences
    padded_target_seqs = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True)
    
    return padded_input_seqs, padded_target_seqs, padded_time_seqs

if __name__ == '__main__':
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autoencoder = BottleneckT5Autoencoder(model_path='thesephist/contra-bottleneck-t5-large-wikipedia', device=device)
    distances_dir = r'C:\Users\angel\Desktop\Chị Huyền\dataset\distances'
    labels_dir = r'C:\Users\angel\Desktop\Chị Huyền\dataset\labels'
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 30
    chunk_size = 384

    # Dataset and DataLoader
    dataset = NPZDataset(distances_dir, labels_dir, autoencoder, chunk_size=chunk_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    config = ModelConfig()
    model = Encoder(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_seq, target_seq, time_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            time_seq = time_seq.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_vectors=input_seq, position_ids=time_seq)

            # Reshape for loss calculation
            # We need to flatten the logits and targets
            logits = logits.view(-1, logits.size(-1))

            # Calculate loss
            loss = criterion(logits, target_seq)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print('Training finished.')

torch.save(model, "trained.pt")