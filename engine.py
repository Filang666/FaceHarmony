import torch
import torch.nn as nn


def weights_init(m):
    """Initializes weights using Xavier Uniform for stable training."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.1)


class PNet(nn.Module):
    """Proposal Network: Fast FCN for initial face candidates."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3), nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3), nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3), nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)  # Face/Non-face
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)  # BBox regression
        self.apply(weights_init)

    def forward(self, x):
        x = self.features(x)
        return torch.softmax(self.conv4_1(x), dim=1), self.conv4_2(x)


class RNet(nn.Module):
    """Refinement Network: Filters P-Net candidates."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3), nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(28, 48, kernel_size=3), nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2), nn.PReLU()
        )
        self.dense = nn.Linear(64 * 3 * 3, 128)
        self.prelu = nn.PReLU()
        self.cls = nn.Linear(128, 2)
        self.bbox = nn.Linear(128, 4)
        self.apply(weights_init)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.prelu(self.dense(x))
        return torch.softmax(self.cls(x), dim=1), self.bbox(x)


class ONet(nn.Module):
    """Output Network: Final face verification and bbox refinement."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3), nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3), nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2), nn.PReLU()
        )
        self.dense = nn.Linear(128 * 3 * 3, 256)
        self.prelu = nn.PReLU()
        self.cls = nn.Linear(256, 2)
        self.bbox = nn.Linear(256, 4)
        self.apply(weights_init)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.prelu(self.dense(x))
        return torch.softmax(self.cls(x), dim=1), self.bbox(x)
