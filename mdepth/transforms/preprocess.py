import math
import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as tf_func
import torch.nn.functional as F


class JointToTensor(object):
    def __call__(self, left, right):
        return tf_func.to_tensor(left), tf_func.to_tensor(right)


class JointCenterCrop(object):
    def __init__(self, size):
        """
        params:
            size (int) : size of the center crop
        """
        self.size = size

    def __call__(self, left, right):
        return (
            tf_func.five_crop(left, self.size)[4],
            tf_func.five_crop(right, self.size)[4],
        )


class JointRandomResizeCrop(object):
    def __init__(self, size, min_scale, max_scale):
        """
        params:
            min_scale (float)
            max_scale (float)
            size (int) : dimension of final crop
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.size = size

    def __call__(self, left, right):
        i, j, h, w = tf.RandomResizedCrop.get_params(
            left, [self.min_scale, self.max_scale], [1.0, 1.0]
        )
        left = tf_func.resized_crop(left, i, j, h, w, self.size)
        right = tf_func.resized_crop(
            right,
            i,
            j,
            h,
            w,
            self.size,
            interpolation=tf.InterpolationMode.NEAREST_EXACT,
        )

        return (left, right)


class JointRandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, left, right):
        r = (
            torch.FloatTensor(
                1,
            )
            .uniform_(0, 1)
            .item()
        )

        if r <= self.p:
            return (tf_func.hflip(left), tf_func.hflip(right))
        else:
            return (left, right)


class JointRandomAugment(object):
    def __init__(
        self, gamma=(0.8, 1.2), brightness=(0.5, 2.0), color=(0.8, 1.2), prob=0.5
    ):
        self.gamma, self.brightness, self.color, self.p = gamma, brightness, color, prob

    def _augment_pair(self, left, right):
        random_gamma = (
            torch.FloatTensor(
                1,
            )
            .uniform_(*self.gamma)
            .item()
        )
        random_bright = (
            torch.FloatTensor(
                1,
            )
            .uniform_(*self.brightness)
            .item()
        )
        random_color_shift = torch.FloatTensor(
            3,
        ).uniform_(*self.color)

        left, right = left**random_gamma, right**random_gamma
        left, right = left * random_bright, right * random_bright

        for i in range(3):
            left[i, :, :] *= random_color_shift[i].item()
            right[i, :, :] *= random_color_shift[i].item()

        left = torch.clamp(left, 0, 1)
        right = torch.clamp(right, 0, 1)

        return (left, right)

    def __call__(self, left, right):
        r = torch.FloatTensor(1, 1).uniform_(0, 1).item()

        if r <= self.p:
            return self._augment_pair(left, right)
        else:
            return (left, right)


class JointNormalize(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, left, right):
        return (
            tf_func.normalize(left, mean=self.mu, std=self.sigma),
            tf_func.normalize(right, mean=self.mu, std=self.sigma),
        )


class JointRoundBy(object):
    def __init__(self, base):
        self.base = float(base)

    def __call__(self, left, right):
        h, w = left.shape[-2:]
        new_h, new_w = int(math.ceil(h / self.base) * self.base), int(
            math.ceil(w / self.base) * self.base
        )
        return (
            F.interpolate(
                left[None], size=(h, w), mode="bilinear", align_corners=True
            ).squeeze(dim=0),
            F.interpolate(
                right[None], size=(h, w), mode="bilinear", align_corners=True
            ).squeeze(dim=0),
        )


class JointResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, left, right):
        return (
            tf.Resize(self.size, antialias=True)(left[None]).squeeze(dim=0),
            tf.Resize(self.size, antialias=True)(right[None]).squeeze(dim=0),
        )


class JointCompose(object):
    def __init__(self, transforms):
        """
        params:
           transforms (list) : list of transforms
        """
        self.transforms = transforms

    def __call__(self, left, right):
        assert left.size == right.size
        for t in self.transforms:
            left, right = t(left, right)
        return left, right
