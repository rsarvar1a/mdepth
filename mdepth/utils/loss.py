import torch

from torch import nn, Tensor
from torch.nn import functional as F


class MonocularLoss(nn.Module):
    """
    Uses the disparity maps to compute a predicted stereo pair from the real stereo pair
    in a bidirectional manner, then computes sublosses on the target-prediction pairs for
    the left and right images separately.
    """

    def __init__(self, *, weight_ssim=0.85, weight_disp=1.0, weight_lr=1.0):
        """
        Creates a new loss with the given subloss weights for the disparity pyramid
        of the given height.
        """
        super().__init__()

        self.weight_ssim = weight_ssim
        self.weight_disp = weight_disp
        self.weight_lr = weight_lr

    def forward(self, disps, stereo_pair):
        """
        Calculates the loss.
        """
        l_target, r_target = stereo_pair

        l_disps = disps[:, 0, :, :].unsqueeze(1)
        r_disps = disps[:, 1, :, :].unsqueeze(1)

        l_est = self._l_image(r_target, l_disps)
        r_est = self._r_image(l_target, r_disps)

        rl_disp = self._l_image(r_est, l_est) 
        lr_disp = self._r_image(l_est, r_est)

        l_smooth = self._smoothness(l_est, l_target)
        r_smooth = self._smoothness(r_est, r_target)

        l_l1 = torch.mean(torch.abs(l_est - l_target))
        r_l1 = torch.mean(torch.abs(r_est - r_target))

        l_ssim = torch.mean(self._ssim(l_est, l_target))
        r_ssim = torch.mean(self._ssim(r_est, r_target))

        l_im_loss = self.weight_ssim * l_ssim + (1 - self.weight_ssim) * l_l1
        r_im_loss = self.weight_ssim * r_ssim + (1 - self.weight_ssim) * r_l1

        l_lr_loss = torch.mean(torch.abs(rl_disp - l_est))
        r_lr_loss = torch.mean(torch.abs(lr_disp - r_est))

        l_disp_loss = torch.mean(torch.abs(l_smooth))
        r_disp_loss = torch.mean(torch.abs(r_smooth))

        im_loss = l_im_loss + r_im_loss
        lr_loss = l_lr_loss + r_lr_loss
        disp_loss = l_disp_loss + r_disp_loss

        loss = im_loss + self.weight_disp * disp_loss + self.weight_lr * lr_loss
        return loss

    def _apply(self, img, disp):
        b, _, h, w = img.shape

        x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
        y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

        x_shifts = disp[:, 0, :, :]
        field = torch.stack((x_base + x_shifts, y_base), dim=3)
        out = F.grid_sample(img, 2 * field - 1, mode="bilinear", padding_mode="zeros")
        return out

    def _grad_x(self, img):
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def _grad_y(self, img):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def _l_image(self, img, disp):
        return self._apply(img, -disp)

    def _r_image(self, img, disp):
        return self._apply(img, disp)

    def _smoothness(self, disp, im):
        d_grad_x = self._grad_x(disp)
        d_grad_y = self._grad_y(disp)

        im_grad_x = self._grad_x(im)
        im_grad_y = self._grad_y(im)

        w_x = torch.exp(-torch.mean(torch.abs(d_grad_x), 1, keepdim=True))
        w_y = torch.exp(-torch.mean(torch.abs(d_grad_y), 1, keepdim=True))

        smooth_x = d_grad_x * w_x
        smooth_y = d_grad_y * w_y

        return torch.abs(smooth_x) + torch.abs(smooth_y)

    def _ssim(self, l, r):
        pool = nn.AvgPool2d(3, 1)
        C1 = 0.01**2
        C2 = 0.03**2

        mu_l = pool(l)
        mu_r = pool(r)
        mu_l_mu_r = mu_l * mu_r
        mu_l_sq = mu_l.pow(2)
        mu_r_sq = mu_r.pow(2)

        s_l = pool(l * l) - mu_l_sq
        s_r = pool(r * r) - mu_r_sq
        s_lr = pool(l * r) - mu_l_mu_r

        ssim_n = (2 * mu_l_mu_r + C1) * (2 * s_lr + C2)
        ssim_d = (mu_l_sq + mu_r_sq + C1) * (s_l + s_r + C2)

        return torch.clamp((1 - (ssim_n / ssim_d)) / 2.0, 0.0, 1.0)
