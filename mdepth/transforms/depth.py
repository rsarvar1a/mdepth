from torch import nn, Tensor
from torch.nn import functional as F


class DisparityToDepth(nn.Module):
    """
    Takes a generated disparity image, and converts it into depth.
    """

    def __init__(self, min_depth, max_depth) -> None:
        """
        Constructs the transformation for the given range.
        """
        super().__init__()

        self.disparity_min = 1.0 / max_depth
        self.disparity_max = 1.0 / min_depth

    def forward(self, x: Tensor) -> Tensor:
        """
        Converts a disparity image into a depth image.
        """
        return 1.0 / (
            self.disparity_min + (self.disparity_max - self.disparity_min) * x
        )
