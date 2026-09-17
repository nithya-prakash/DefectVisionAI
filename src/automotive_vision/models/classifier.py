"""Transfer-learning defect classifier.

Backbone choice: ResNet18 (ImageNet-pretrained), chosen over EfficientNet/ConvNeXt for
this project because (a) the labeled pool is tiny (~80 training images across 5
classes, see data/README.md), so a larger backbone would only increase overfitting
risk without more data to justify it, and (b) it is fast enough to fine-tune and run
repeatedly on CPU/MPS, which matters for the active-learning simulation's many retrain
cycles (src/automotive_vision/active_learning/simulation.py).

Fine-tuning strategy: only layer4 and the final classification head are unfrozen; layer1-3
stay frozen at their ImageNet weights. This is a partial fine-tune, not full fine-tuning,
specifically because of the tiny training set — fine-tuning the whole network on ~80
images invites overfitting.
"""

import torchvision.models as tv_models
from torch import nn


class DefectClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_until_layer4: bool = True, dropout_p: float = 0.3):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_until_layer4:
            for name, param in backbone.named_parameters():
                if not name.startswith(("layer4", "fc")):
                    param.requires_grad = False

        in_features = backbone.fc.in_features
        # Dropout before the head doubles as regularization for a ~80-image training
        # set and as the stochastic component for Monte Carlo dropout at inference.
        backbone.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
