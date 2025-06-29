import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        embedded = self.dropout(src)
        # embedded shape: (batch_size, seq_len, input_dim)

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs shape: (batch_size, seq_len, hidden_dim * num_directions) - assuming bidirectional=False
        # hidden shape: (n_layers * num_directions, batch_size, hidden_dim)
        # cell shape: (n_layers * num_directions, batch_size, hidden_dim)

        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, n_heads, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        # MultiheadAttention expects input shape (seq_len, batch_size, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=False)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim, output_dim) # Decoder RNN output + context vector

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input shape: (batch_size, 1) - previous token index
        # hidden shape: (n_layers, batch_size, hidden_dim)
        # cell shape: (n_layers, batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)

        # Embed the input token
        input = input.unsqueeze(1) # Add sequence length dimension (1)
        embedded = self.dropout(self.embedding(input))
        # embedded shape: (batch_size, 1, embed_dim)

        # Pass through the decoder RNN
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output shape: (batch_size, 1, hidden_dim)
        # hidden shape: (n_layers, batch_size, hidden_dim)
        # cell shape: (n_layers, batch_size, hidden_dim)

        # Prepare query, key, value for multihead attention
        # Query is the top layer decoder hidden state
        query = hidden[-1].unsqueeze(0) # shape: (1, batch_size, hidden_dim)
        # Key and Value are the encoder outputs
        key_value = encoder_outputs.transpose(0, 1) # shape: (src_seq_len, batch_size, hidden_dim)

        # Calculate attention
        attn_output, attn_output_weights = self.multihead_attn(query, key_value, key_value)
        # attn_output shape: (1, batch_size, hidden_dim)
        # attn_output_weights shape: (batch_size, 1, src_seq_len) - attention weights for the single query position

        # Squeeze the sequence length dimension from attention output
        context = attn_output.squeeze(0) # shape: (batch_size, hidden_dim)

        # Concatenate decoder RNN output and context vector for final prediction
        # Squeeze the sequence length dimension from decoder RNN output
        prediction = self.fc_out(torch.cat((output.squeeze(1), context), dim=1))
        # prediction shape: (batch_size, output_dim)

        return prediction, hidden, cell, attn_output_weights.squeeze(1) # Return weights shape (batch_size, src_seq_len)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src shape: (batch_size, src_seq_len, input_dim)
        # trg shape: (batch_size, trg_seq_len) - target token indices

        batch_size = src.shape[0]
        trg_seq_len = trg.shape[1]
        output_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_seq_len, output_vocab_size).to(self.device)

        # Encoder forward pass
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input to the decoder is the <sos> token (assuming index 0 is <sos>)
        decoder_input = trg[:, 0] # shape: (batch_size)

        for t in range(1, trg_seq_len):
            # Decoder forward pass
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)

            # Store prediction
            outputs[:, t, :] = output

            # Decide whether to use teacher forcing (per sample in batch)
            teacher_force = (torch.rand(batch_size, device=trg.device) < teacher_forcing_ratio)

            # Get the highest predicted token index
            top1 = output.argmax(1)

            # Use ground truth token as next input if teacher forcing, otherwise use predicted token (per sample)
            decoder_input = torch.where(teacher_force, trg[:, t], top1)

        return outputs

    def translate(self, src, max_len=5):
        # src shape: (batch_size, src_seq_len, input_dim)

        batch_size = src.shape[0]

        # Encoder forward pass
        encoder_outputs, hidden, cell = self.encoder(src)

        # Tensor to store output tokens
        # Start with <sos> token index (assuming index 0 is <sos>)
        trg_tokens = [torch.zeros(batch_size, dtype=torch.long).fill_(0).to(self.device)]

        # First input to the decoder is the <sos> token index
        decoder_input = trg_tokens[0] # shape: (batch_size)

        for t in range(1, max_len):
            # Decoder forward pass
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)

            # Get the highest predicted token index
            top1 = output.argmax(1)

            # Store the predicted token
            trg_tokens.append(top1)

            # Use the predicted token as the next input
            decoder_input = top1

            # Stop if all sequences in the batch predict the <eos> token (assuming index 1 is <eos>)
            # if all(token == 1 for token in top1):
            #     break

        # Stack the predicted tokens
        trg_tokens = torch.stack(trg_tokens, dim=1)
        # trg_tokens shape: (batch_size, predicted_seq_len)

        return trg_tokens

# Example Usage (requires defining dimensions and instantiating)
# input_dim = ... # Dimension of your input vectors
# output_vocab_size = ... # Number of unique tokens in your output vocabulary
# embed_dim = ... # Dimension for token embeddings (can be same as hidden_dim)
# hidden_dim = 256
# n_layers = 2
# n_heads = 2 # Number of attention heads
# dropout = 0.5
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# encoder = Encoder(input_dim, hidden_dim, n_layers, dropout)
# decoder = Decoder(output_vocab_size, embed_dim, hidden_dim, n_layers, n_heads, dropout)

# model = Seq2Seq(encoder, decoder, device).to(device)

# To train, you would need:
# 1. Data loading and preprocessing (sequences of vectors and corresponding token sequences)
# 2. Loss function (e.g., CrossEntropyLoss)
# 3. Optimizer (e.g., Adam)
# 4. Training loop with forward and backward passes

# For real-time inference, you would load a trained model and use the .translate() method.
# You would also need to handle the real-time input vector sequence collection.
