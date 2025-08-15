import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18AU(nn.Module):
    def __init__(self, num_classes=6, pretrained=False, au_feature_dim=12):
        super(ResNet18AU, self).__init__()

        self.frame_branch = models.resnet18(pretrained=pretrained)
        in_features = self.frame_branch.fc.in_features  # 512 for ResNet18
        self.frame_branch.fc = nn.Identity()

        # AU branch with correct dimensions to match checkpoint
        self.au_branch = nn.Sequential(
            nn.Linear(au_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 128),  # This should output 128 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Fusion classifier with correct dimensions
        fusion_in_features = in_features + 128  # 512 + 128 = 640
        self.fc = nn.Sequential(
            nn.Linear(fusion_in_features, 256),  # 640 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)  # 256 -> 6 (or num_classes)
        )

    def forward(self, x, au_features=None):
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
        cnn_features = self.frame_branch(x)

        if au_features is None:
            au_features = torch.zeros(cnn_features.size(0), 12, device=cnn_features.device)

        au_features = self.au_branch(au_features)

        fused = torch.cat((cnn_features, au_features), dim=1)
        out = self.fc(fused)

        if 't' in locals():
            out = out.view(b, t, -1).mean(dim=1)

        return out

if __name__ == "__main__":
    model = ResNet18AU(num_classes=6, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_au = torch.randn(2, 12)
    output = model(dummy_input, dummy_au)
    print("Output shape:", output.shape)  # should be (2, 6)
