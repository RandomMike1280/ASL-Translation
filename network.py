import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Model(nn.Module):
    def __init__(self, latent_dim=None, vocab_size=None):
        super(Model, self).__init__()
        self.latent_dim = latent_dim if latent_dim else 32
        self.mlp = MLP(input_size=1106, hidden_size=128, output_size=self.latent_dim)
        self.lstm_encoder = nn.LSTM(input_size=self.latent_dim, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.out = nn.Linear(64, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, pre_encoder, pre_lstm):
        x = self.mlp(x)
        x = x.unsqueeze(0)
        x, pre_encoder = self.lstm_encoder(x, pre_encoder)
        x, pre_lstm = self.lstm(x, pre_lstm)
        x = self.out(x)
        x = self.softmax(x)
        x = x.squeeze(0)
        return x, pre_encoder, pre_lstm
        
if __name__ == "__main__":
    model = Model(vocab_size=10)
    print("First time step")
    x = torch.randn(1, 1106)
    pre_encoder = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))
    pre_lstm = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))
    x, pre_encoder, pre_lstm = model(x, pre_encoder, pre_lstm)
    print("x shape: ", x)
    print("pre_encoder[0] shape: ", pre_encoder[0].shape)
    print("pre_encoder[1] shape: ", pre_encoder[1].shape)
    print("pre_lstm[0] shape: ", pre_lstm[0].shape)
    print("pre_lstm[1] shape: ", pre_lstm[1].shape)
    print("Second time step")
    x = torch.randn(1, 1106)
    x, pre_encoder, pre_lstm = model(x, pre_encoder, pre_lstm)
    print("x shape: ", x)
    print("pre_encoder[0] shape: ", pre_encoder[0].shape)
    print("pre_encoder[1] shape: ", pre_encoder[1].shape)
    print("pre_lstm[0] shape: ", pre_lstm[0].shape)
    print("pre_lstm[1] shape: ", pre_lstm[1].shape)