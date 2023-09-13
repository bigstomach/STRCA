import torch.nn as nn

class GRUClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out


def build_grucls(input_size, hidden_size, num_layers, num_classes):
    return GRUClassification(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes
    )
