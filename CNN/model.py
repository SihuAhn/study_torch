import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = CNN()
print(model)

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()