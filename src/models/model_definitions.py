import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        return self.layers(x)


class SimpleCNN(nn.Module):  # Простенькая сверточная
    def __init__(self, num_classes=29):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AdvancedCNN(nn.Module):  # Крутая сверточная
    def __init__(self, num_classes=29):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))