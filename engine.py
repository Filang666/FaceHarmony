import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class PNet(nn.Module):
    """Lightweight Proposal Network for initial face detection."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3), nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3), nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3), nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        return torch.softmax(self.conv4_1(x), dim=1), self.conv4_2(x)


class MobileFaceLandmarker(nn.Module):
    """MobileNetV2-based regressor for 68 facial landmarks."""
    def __init__(self):
        super().__init__()
        # Use a slimmed MobileNetV2
        mnet = mobilenet_v2(pretrained=False)
        self.features = mnet.features
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 68 * 2)  # 68 points (x, y)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


class LooksmaxScorer(nn.Module):
    """Deep scoring for facial symmetry and harmony."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(68 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x) * 10
