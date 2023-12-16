import os

from PIL import Image
from torch.utils.data import Dataset


class KittiDataset(Dataset):
    """
    Args:

        path: location of kitti dataset
        mode: train or test mode
        transforms: transforms to be applied on dataset
    """

    def __init__(self, path, mode="train", transforms=None):
        # get list of files of left images
        left_imgs_loc = os.path.join(path, "image_2/")
        self.left_imgs = sorted(
            [os.path.join(left_imgs_loc, f) for f in os.listdir(left_imgs_loc)]
        )

        if mode == "train":
            # get list of files of right images
            right_imgs_loc = os.path.join(path, "image_3/")
            self.right_imgs = sorted(
                [os.path.join(right_imgs_loc, f) for f in os.listdir(right_imgs_loc)]
            )
            assert len(self.left_imgs) == len(self.right_imgs)

        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.left_imgs)

    def __getitem__(self, index):
        left = Image.open(self.left_imgs[index])
        if self.mode == "train":
            right = Image.open(self.right_imgs[index])
            return self.transforms((left, right)) if self.transforms else (left, right)
        else:
            return self.transforms(left) if self.transforms else left
