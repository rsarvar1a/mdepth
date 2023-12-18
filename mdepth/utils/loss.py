
import torch

from torch import nn, Tensor
from torch.nn import functional as F


class MonocularLoss(nn.Module):
    """
    Uses the disparity maps to compute a predicted stereo pair from the real stereo pair
    in a bidirectional manner, then computes sublosses on the target-prediction pairs for
    the left and right images separately.
    """

    def __init__(self, *, n=4, weight_ssim=0.85, weight_disp=1.0, weight_lr=1.0):
        """
        Creates a new loss with the given subloss weights for the disparity pyramid
        of the given height.
        """
        super().__init__()

        self.n = n
        self.weight_ssim = weight_ssim
        self.weight_disp = weight_disp
        self.weight_lr = weight_lr

    def _pyramid(self, img):
        """
        Generate a pyramid of images, scaled by powers of two.
        """
        h, w = img.shape[-2:]
        
        imgs = [img]
        for i in range(self.n - 1):
            ratio = 2 ** (i + 1)
            imgs.append(
                F.interpolate(
                    img,
                    size=(int(h // ratio), int(w // ratio)), 
                    mode='bilinear',
                    align_corners=True
                )
            )
        
        return imgs

    def _gradient_x(self, img):
        """
        Calculates the first difference of x.
        """
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def _gradient_y(self, img):
        """
        Calculates the first difference of y.
        """
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def _apply(self, img, disp):
        """
        Generates a synthetic stereo image by applying the disparity map to the reference image.
        """
        b, _, h, w = img.size()

        x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
        y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

        x_shifts = disp[:, 0, :, :]
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros', align_corners=True)

        return output

    def _l_image(self, img, disp):
        """
        Given a reference image from the right, applies a disparity map in reverse
        to synthesize a left image.
        """
        return self._apply(img, -disp)

    def _r_image(self, img, disp):
        """
        Given a reference image from the left, applies a disparity map to synthesize
        a right image.
        """
        return self._apply(img, disp)

    def _ssim(self, l, r):
        """
        Returns the SSIM loss of the two iamges.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        pool = nn.AvgPool2d(3, 1)

        mu_l = pool(l)
        mu_r = pool(r)
        mu_l_mu_r = mu_l * mu_r
        mu_l_sq = mu_l.pow(2)
        mu_r_sq = mu_r.pow(2)

        sigma_l = pool(l * l) - mu_l_sq
        sigma_r = pool(r * r) - mu_r_sq
        sigma_lr = pool(l * r) - mu_l_mu_r

        ssim_n = (2 * mu_l_mu_r + C1) * (2 * sigma_lr + C2)
        ssim_d = (mu_l_sq + mu_r_sq + C1) * (sigma_l + sigma_r + C2)

        return torch.clamp((1 - (ssim_n / ssim_d)) / 2, 0, 1)

    def _smoothness(self, disp, pyramid):
        """
        Computes the smoothness loss.
        """
        disp_gradients_x = [self._gradient_x(d) for d in disp]
        disp_gradients_y = [self._gradient_y(d) for d in disp]

        image_gradients_x = [self._gradient_x(img) for img in pyramid]
        image_gradients_y = [self._gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(self.n)]

    def forward(self, disps, stereo):
        """
        Returns the monocular depth loss, as computed by taking the output disparities,
        generating the synthetic stereo pair, and comparing them to the true stereo pair.
        """
        l, r = stereo
        l_pyramid = self._pyramid(l)
        r_pyramid = self._pyramid(r)

        l_disp_est = [d[:, 0, :, :].unsqueeze(1) for d in disps]
        r_disp_est = [d[:, 1, :, :].unsqueeze(1) for d in disps]

        l_est = [self._l_image(r_pyramid[i], l_disp_est[i]) for i in range(self.n)]
        r_est = [self._r_image(l_pyramid[i], r_disp_est[i]) for i in range(self.n)]

        rl_disp = [self._l_image(r_disp_est[i], l_disp_est[i]) for i in range(self.n)]
        lr_disp = [self._r_image(l_disp_est[i], r_disp_est[i]) for i in range(self.n)]

        l_smoothness = self._smoothness(l_disp_est, l_pyramid)
        r_smoothness = self._smoothness(r_disp_est, r_pyramid)

        l_l1_loss = [torch.mean(torch.abs(l_est[i] - l_pyramid[i])) for i in range(self.n)]
        r_l1_loss = [torch.mean(torch.abs(r_est[i] - r_pyramid[i])) for i in range(self.n)]

        l_ssim_loss = [torch.mean(self._ssim(l_est[i], l_pyramid[i])) for i in range(self.n)]
        r_ssim_loss = [torch.mean(self._ssim(r_est[i], r_pyramid[i])) for i in range(self.n)]

        l_image_loss = [self.weight_ssim * l_ssim_loss[i] + (1 - self.weight_ssim) * l_l1_loss[i] for i in range(self.n)]
        r_image_loss = [self.weight_ssim * r_ssim_loss[i] + (1 - self.weight_ssim) * r_l1_loss[i] for i in range(self.n)]
        
        l_lr_loss = [torch.mean(torch.abs(rl_disp[i] - l_disp_est[i])) for i in range(self.n)]
        r_lr_loss = [torch.mean(torch.abs(lr_disp[i] - r_disp_est[i])) for i in range(self.n)]

        l_disp_loss = [torch.mean(torch.abs(l_smoothness[i])) / 2 ** i for i in range(self.n)]
        r_disp_loss = [torch.mean(torch.abs(r_smoothness[i])) / 2 ** i for i in range(self.n)]
        
        image_loss = sum(l_image_loss + r_image_loss)
        lr_loss = sum(l_lr_loss + r_lr_loss)
        disp_loss = sum(l_disp_loss + r_disp_loss)

        loss = image_loss + self.weight_disp * disp_loss + self.weight_lr * lr_loss
        return loss

        