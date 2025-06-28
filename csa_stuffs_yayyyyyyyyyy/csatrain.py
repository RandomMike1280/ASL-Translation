import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob
from collections import Counter
from torch.nn.utils.rnn import pad_sequence # Import pad_sequence

# Assuming your translator_model.py is in the same directory
from translator_model import Encoder, Decoder, Seq2Seq

# Define special tokens and their indices
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
NONE_TOKEN = '<none>' # Token for frames with no specific label in CSV

# Define the maximum target sequence length (5 tokens + SOS + EOS)
MAX_TARGET_LEN = 5 + 2 # SOS + 5 tokens + EOS

class ASLTranslationDataset(Dataset):
    def __init__(self, data_dir, vocab=None):
        self.data_dir = data_dir
        self.distances_dir = os.path.join(data_dir, 'distances')
        self.labels_dir = os.path.join(data_dir, 'labels')

        # Get list of file prefixes (e.g., '1', '2', ...)
        self.file_prefixes = [os.path.splitext(os.path.basename(f))[0]
                              for f in glob.glob(os.path.join(self.distances_dir, '*.npz'))]
        self.file_prefixes.sort() # Ensure consistent order

        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        self.sos_idx = self.token_to_idx[SOS_TOKEN]
        self.eos_idx = self.token_to_idx[EOS_TOKEN]
        self.pad_idx = self.token_to_idx[PAD_TOKEN]
        self.none_idx = self.token_to_idx[NONE_TOKEN]


    def build_vocab(self):
        all_labels = set()
        for prefix in self.file_prefixes:
            csv_path = os.path.join(self.labels_dir, f'{prefix}.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Add labels from the 'label' column, excluding NaN if any
                all_labels.update(df['label'].dropna().unique())

        # Include special tokens
        vocab = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, NONE_TOKEN] + sorted(list(all_labels))
        return vocab

    def __len__(self):
        return len(self.file_prefixes)

    def __getitem__(self, idx):
        prefix = self.file_prefixes[idx]
        distances_path = os.path.join(self.distances_dir, f'{prefix}.npz')
        labels_path = os.path.join(self.labels_dir, f'{prefix}.csv')

        # Load input sequence (vectors)
        # Assuming the numpy array is stored under the key 'arr_0' or 'distances'
        try:
            input_sequence = np.load(distances_path)['arr_0']
        except KeyError:
            input_sequence = np.load(distances_path)['distances']

        input_sequence = torch.FloatTensor(input_sequence) # Convert to tensor

        # Load and process target sequence (tokens)
        target_tokens = [SOS_TOKEN] # Start with SOS token
        if os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            # Extract unique non-NONE labels, sorted by first appearance frame
            # Assuming 'frame_number' column exists and is sorted
            unique_labels = df[df['label'] != NONE_TOKEN]['label'].unique().tolist()

            # Take up to 5 unique labels
            selected_labels = unique_labels[:5]
            target_tokens.extend(selected_labels)

        # Add EOS token and pad
        target_tokens.append(EOS_TOKEN)
        while len(target_tokens) < MAX_TARGET_LEN:
            target_tokens.append(PAD_TOKEN)

        # Convert tokens to indices
        target_indices = [self.token_to_idx[token] for token in target_tokens]
        target_indices = torch.LongTensor(target_indices)

        return input_sequence, target_indices

# Custom collate function to handle variable-length input sequences
def collate_batch(batch):
    # print("Using custom collate_batch") # Add this line to check if the function is called
    # batch is a list of tuples: [(input_sequence1, target_indices1), (input_sequence2, target_indices2), ...]
    input_sequences, target_indices = zip(*batch)

    # Pad the input sequences to the maximum length in the batch
    # pad_sequence expects a list of tensors, and pads them to the length of the longest tensor
    # batch_first=True makes the output shape (batch_size, seq_len, feature_dim)
    padded_input_sequences = pad_sequence(input_sequences, batch_first=True, padding_value=0) # Assuming 0 is a safe padding value for vectors

    # Target indices are already padded to MAX_TARGET_LEN in __getitem__, just stack them
    target_indices = torch.stack(target_indices, dim=0)

    return padded_input_sequences, target_indices


# --- Training Setup ---

# Define model parameters (example values, adjust as needed)
INPUT_DIM = 1106 # Dimension of your input vectors
# OUTPUT_VOCAB_SIZE will be determined by the dataset's vocabulary size
EMBED_DIM = 256 # Dimension for token embeddings
HIDDEN_DIM = 256
N_LAYERS = 4
N_HEADS = 4
DROPOUT = 0.5

# Define training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIRECTORY = 'dataset' # Path to your dataset folder

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    print('TRAININGGG')
    # Create dataset and dataloader
    dataset = ASLTranslationDataset(DATA_DIRECTORY)
    # Pass the custom collate function to the DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    # Determine output vocabulary size from the dataset
    OUTPUT_VOCAB_SIZE = len(dataset.vocab)

    # Instantiate the model
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Define loss function and optimizer
    # Ignore padding index in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        

    # --- Training Loop ---

    print(f"Starting training on {device}")

    for checkpoint_idx in range(9999999999999999999):
        for epoch in range(NUM_EPOCHS):
            model.train() # Set model to training mode
            epoch_loss = 0

            for i, (src, trg) in enumerate(dataloader):
                src, trg = src.to(device), trg.to(device)

                optimizer.zero_grad()

                # Forward pass
                # The model's forward method expects target sequence for teacher forcing
                # outputs shape: (batch_size, trg_seq_len, output_vocab_size)
                outputs = model(src, trg)

                # Reshape outputs and target for loss calculation
                # Ignore the first token (<sos>) in the target and outputs for loss
                output_dim = outputs.shape[-1]
                outputs = outputs[:, 1:].reshape(-1, output_dim) # shape: (batch_size * (trg_seq_len - 1), output_vocab_size)
                trg = trg[:, 1:].reshape(-1) # shape: (batch_size * (trg_seq_len - 1))

                loss = criterion(outputs, trg)

                # Backward pass and optimize
                loss.backward()
                # Optional: Clip gradients to prevent exploding gradients
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                epoch_loss += loss.item()

                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {epoch_loss / len(dataloader):.4f}')


        # Optional: Save the trained model
        checkpoint_path = fr'csa_stuffs_yayyyyyyyyyy\checkpoints\{checkpoint_idx}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print("Model saved to " + checkpoint_path)
