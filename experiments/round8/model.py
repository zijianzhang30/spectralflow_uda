"""DCRN convolutional branches before attention; single-domain classifier."""
import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self, bands=48, classes=7):
        super().__init__()
        layers = {
            1: (1, 24, (7, 1, 1), (2, 1, 1), 0),
            2: (24, 24, (7, 1, 1), 1, (3, 0, 0)),
            3: (24, 24, (7, 1, 1), 1, (3, 0, 0)),
            4: (24, 192, ((bands - 7) // 2 + 1, 1, 1), 1, 0),
            5: (1, 24, (bands, 1, 1), 1, 0),
            6: (24, 24, (1, 3, 3), 1, (0, 1, 1)),
            7: (24, 96, (1, 3, 3), 1, (0, 1, 1)),
        }
        for i, (cin, cout, kernel, stride, padding) in layers.items():
            setattr(self, f"conv{i}", nn.Conv3d(cin, cout, kernel, stride, padding))
            setattr(self, f"bn{i}", nn.BatchNorm3d(cout))
        self.conv8 = nn.Conv3d(24, 96, 1)
        self.classifier = nn.Linear(288, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def features(self, x):
        x = x.unsqueeze(1)
        spectral = self.bn1(self.conv1(x)).relu()
        residual = spectral
        spectral = self.bn2(self.conv2(spectral)).relu()
        spectral = self.bn3(self.conv3(spectral) + residual).relu()
        spectral = self.bn4(self.conv4(spectral)).relu()
        spatial = self.bn5(self.conv5(x)).relu()
        residual = self.conv8(spatial)
        spatial = self.bn6(self.conv6(spatial)).relu()
        spatial = self.bn7(self.conv7(spatial) + residual).relu()
        # Explicit global reduction avoids the nondeterministic AvgPool3d backward.
        return torch.cat((spectral, spatial), dim=1).mean(dim=(2, 3, 4))

    def forward(self, x):
        z = self.features(x)
        return z, self.classifier(z)
