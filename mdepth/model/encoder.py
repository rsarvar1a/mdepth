from torch import nn, rand, Tensor
from torch.nn import functional as F
from torchvision.models import resnet101, ResNet101_Weights


class ImageEncoder(nn.Module):
    """
    Takes in an tensor of RGB images, and outputs a list of feature tensors.
    """

    def __init__(self) -> None:
        """
        Constructs a new ImageEncoder by scrapping a ResNet101.
        """
        super().__init__()

        # Build the model by decomposing ResNet; this is necessary because we'd
        # like to hook into the model between layer calls and save intermediate results.
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.conv1  = resnet.conv1
        self.norm1  = resnet.bn1
        self.relu1  = resnet.relu
        self.pool1  = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Compute the number of channels in each layer's output, so we can use them
        # as parameters while constructing the network heads.
        features = self.forward(rand([1, 3, 64, 64]))
        self.channels = [feature.shape[1] for feature in features]

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Returns a list of feature images.
        """
        x = self.pool1(self.relu1(self.norm1(self.conv1(x))))
        feature_maps: list[Tensor] = []

        x = self.layer1(x)
        feature_maps.append(x)
        x = self.layer2(x)
        feature_maps.append(x)
        x = self.layer3(x)
        feature_maps.append(x)
        # x = self.layer4(x)
        # feature_maps.append(x)

        return feature_maps
