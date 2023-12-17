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
        
        l_path = os.path.join(path, "data_object_image_2", f"{mode}ing")
        r_path = os.path.join(path, "data_object_image_3", f"{mode}ing")
        self.l_images = sorted([os.path.join(l_path, f) for f in os.listdir(l_path)])
        self.r_images = sorted([os.path.join(r_path, f) for f in os.listdir(r_path)])
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        
        return len(self.l_images)

    def __getitem__(self, index):
        
        l = Image.open(self.l_images[index])
        r = Image.open(self.r_images[index])
        
        if self.transforms:
            for t in self.transforms:
                l, r = t(l, r)
        return l, r
