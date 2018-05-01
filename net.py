import torch.nn as nn


class DNCNN(nn.Module):
    def __init__(self, num_channels, num_of_layers=15):
        super(DNCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        first_layer = self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True))
        layers.append(first_layer)
        for _ in range(num_of_layers-2):
            second_layer = self.cnn1 = nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2))
            layers.append(second_layer)

        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.DNCNN = nn.ModuleList(layers)

    def forward(self, x):
        out = self.DNCNN[0](x)
        for i in range(len(self.DNCNN) - 2):
            out = out + self.DNCNN[i + 1](out)
        out = self.DNCNN[-1](out)
        return out
