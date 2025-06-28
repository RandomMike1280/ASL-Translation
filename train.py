import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from network import Model

class Tokenizer:
    def __init__(self, tokens=None):
        self.token2id = {}
        self.id2token = {}
        if tokens:
            for token in tokens:
                self.add_token(token)
        if '<None>' not in self.token2id:
            self.add_token('<None>')

    def add_token(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def encode(self, tokens):
        return [self.token2id.get(tok, self.token2id['<None>']) for tok in tokens]

    def decode(self, ids):
        return [self.id2token.get(i, '<None>') for i in ids]

class NPZDataset(Dataset):
    def __init__(self, distances_dir, labels_dir, tokenizer=None, transform=None):
        self.samples = []
        self.tokenizer = tokenizer or Tokenizer()
        self.transform = transform
        # pre-seed tokenizer with all dataset labels
        unique_tokens = set()
        for f in os.listdir(labels_dir):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(labels_dir, f))
                unique_tokens.update(df.iloc[:,1].astype(str).tolist())
        unique_tokens.add('<None>')
        for tok in unique_tokens:
            self.tokenizer.add_token(tok)
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

            label_ids = self.tokenizer.encode(labels_list)

            # tensorize sequence
            seq_tensor = torch.tensor(arrays, dtype=torch.float)
            if self.transform:
                seq_tensor = self.transform(seq_tensor)
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            self.samples.append((seq_tensor, label_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, labels = self.samples[idx]
        # prepare autoregressive inputs and targets
        # input: all but last timestep; target: all but first label
        input_seq = seq[:-1]
        target_seq = labels[1:]
        # generate time indices for each input timestep
        time_seq = torch.arange(input_seq.size(0), dtype=torch.long)
        return input_seq, target_seq, time_seq

if __name__ == '__main__':
    # initialize tokenizer and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    dataset = NPZDataset('dataset/distances', 'dataset/labels', tokenizer=tokenizer)
    print(f"Vocab size: {len(tokenizer.token2id)}")
    print(f"Tokens: {tokenizer.id2token}")
    # sequential timestep iterator without padding
    class SequenceBatchIterator:
        def __init__(self, samples, batch_size):
            self.samples = samples
            self.batch_size = batch_size

        def __iter__(self):
            for i in range(0, len(self.samples), self.batch_size):
                batch = self.samples[i:i+self.batch_size]
                seqs, labs = zip(*batch)
                t = 0
                while True:
                    # only keep sequences with a next label
                    alive = [j for j, s in enumerate(seqs) if t+1 < s.size(0)]
                    if not alive:
                        break
                    x_t = torch.stack([seqs[j][t] for j in alive], dim=0)
                    y_t = torch.stack([labs[j][t+1] for j in alive], dim=0)
                    yield x_t, y_t
                    t += 1

    # use sequential timestep iterator
    model = Model(vocab_size=len(tokenizer.token2id))
    seq_loader = SequenceBatchIterator(dataset.samples, batch_size=512)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in range(10):
        pre_encoder = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))
        pre_lstm = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))
        print(f"Epoch {epoch}")
        for x_t, y_t in seq_loader:
            x_t, y_t = x_t.to(device), y_t.to(device)
            optimizer.zero_grad()
            y_pred, pre_encoder, pre_lstm = model(x_t, pre_encoder, pre_lstm)
        loss = criterion(y_pred.view(-1, y_pred.size(-1)), y_t.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
    




