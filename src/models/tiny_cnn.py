import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, n_classes=22):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)
