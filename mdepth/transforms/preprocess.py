import torchvision.transforms as tf
import torchvision.transforms.functional as tf_func


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


class JointNormalize(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, left, right):
        return (
            tf_func.normalize(left, mean=self.mu, std=self.sigma),
            tf_func.normalize(right, mean=self.mu, std=self.sigma),
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
