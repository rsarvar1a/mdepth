import torch

from torch import nn, Tensor
from torch.nn import functional as F


class MonocularLoss(nn.Module):
    """
    Uses the disparity maps to compute a predicted stereo pair from the real stereo pair
    in a bidirectional manner, then computes sublosses on the target-prediction pairs for
    the left and right images separately.
    """

    def __init__(self, *, n, weight_ssim=0.85, weight_disp=1.0, weight_lr=1.0):
        """
        Creates a new loss with the given subloss weights for the disparity pyramid
        of the given height.
        """
        super().__init__()

        self.n = n

        self.weight_ssim = weight_ssim
        self.weight_disp = weight_disp
        self.weight_lr = weight_lr

    def forward(self, disps, stereo_pair):
        """
        Calculates the loss.
        """
        l_target, r_target = stereo_pair

        l_stack = self._pyramid(l_target)
        r_stack = self._pyramid(r_target)

        l_disps = [d[:, 0, :, :].unsqueeze(1) for d in disps]
        r_disps = [d[:, 1, :, :].unsqueeze(1) for d in disps]

        l_est = [self._l_image(r_stack[i], l_disps[i]) for i in range(self.n)]
        r_est = [self._r_image(l_stack[i], r_disps[i]) for i in range(self.n)]

        rl_disp = [self._l_image(r_est[i], l_est[i]) for i in range(self.n)]
        lr_disp = [self._r_image(l_est[i], r_est[i]) for i in range(self.n)]

        l_smooth = self._smoothness(l_est, l_stack)
        r_smooth = self._smoothness(r_est, r_stack)

        l_l1 = [torch.mean(torch.abs(l_est[i] - l_stack[i])) for i in range(self.n)]
        r_l1 = [torch.mean(torch.abs(r_est[i] - r_stack[i])) for i in range(self.n)]

        l_ssim = [torch.mean(self._ssim(l_est[i], l_stack[i])) for i in range(self.n)]
        r_ssim = [torch.mean(self._ssim(r_est[i], r_stack[i])) for i in range(self.n)]

        l_im_loss = [
            self.weight_ssim * l_ssim[i] + (1 - self.weight_ssim) * l_l1[i]
            for i in range(self.n)
        ]
        r_im_loss = [
            self.weight_ssim * r_ssim[i] + (1 - self.weight_ssim) * r_l1[i]
            for i in range(self.n)
        ]

        l_lr_loss = [
            torch.mean(torch.abs(rl_disp[i] - l_est[i])) for i in range(self.n)
        ]
        r_lr_loss = [
            torch.mean(torch.abs(lr_disp[i] - r_est[i])) for i in range(self.n)
        ]

        l_disp_loss = [
            torch.mean(torch.abs(l_smooth[i])) / 2.0**i for i in range(self.n)
        ]
        r_disp_loss = [
            torch.mean(torch.abs(r_smooth[i])) / 2.0**i for i in range(self.n)
        ]

        im_loss = sum(l_im_loss + r_im_loss)
        lr_loss = sum(l_lr_loss + r_lr_loss)
        disp_loss = sum(l_disp_loss + r_disp_loss)

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

    def _pyramid(self, img):
        imgs = [img]
        _, _, h, w = img.shape
        for i in range(self.n - 1):
            r = 2 ** (i - 1)
            imgs.append(
                F.interpolate(
                    img, size=(int(h // r), int(w // r)), mode="bilinear", align_corners=True
                )
            )
        return imgs

    def _smoothness(self, disp, stack):
        d_grad_x = [self._grad_x(d) for d in disp]
        d_grad_y = [self._grad_y(d) for d in disp]

        im_grad_x = [self._grad_x(im) for im in stack]
        im_grad_y = [self._grad_y(im) for im in stack]

        w_x = [
            torch.exp(-torch.mean(torch.abs(grad), 1, keepdim=True))
            for grad in im_grad_x
        ]
        w_y = [
            torch.exp(-torch.mean(torch.abs(grad), 1, keepdim=True))
            for grad in im_grad_y
        ]

        smooth_x = [d_grad_x[i] * w_x[i] for i in range(self.n)]
        smooth_y = [d_grad_y[i] * w_y[i] for i in range(self.n)]

        return [torch.abs(smooth_x[i]) + torch.abs(smooth_y[i]) for i in range(self.n)]

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
