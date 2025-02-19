import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    """
    Neural network model based on ResNet-50 with a custom feature extractor and projection head.

    Attributes:
        f (nn.Sequential): Feature extractor using a modified ResNet-50 backbone.
        g (nn.Sequential): Projection head consisting of linear layers, batch normalization, and ReLU activation.
    """

    def __init__(self, feature_dim: int = 128) -> None:
        """
        Initializes the Model.

        Args:
            feature_dim (int): Dimensionality of the feature representation. Default is 128.
        """
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            # Exclude fully connected and max pooling layers from ResNet-50
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                # Enable gradient computation for layers 2, 3, and 4
                if name in ['layer2', 'layer3', 'layer4']:
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    for param in module.parameters():
                        param.requires_grad = False

                self.f.append(module)

        # Feature extractor
        self.f = nn.Sequential(*self.f)

        # Projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Normalized feature representation (before projection head).
                - Normalized output from the projection head.
        """
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



