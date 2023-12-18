import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import tqdm

from .transforms import DisparityToDepth


def test(model, *, dataloader, device, loss, num_samples):
    model.eval()

    indices = random.sample(list(range(len(dataloader))), num_samples)
    samples = []
    total_loss = 0.0

    with torch.no_grad():
        for batch, data in tqdm.tqdm(
            enumerate(dataloader), unit="batch", total=len(dataloader)
        ):
            l_image, r_image = data[0].to(device), data[1].to(device)  # 1, 3, h, w
            disparities = model(l_image)                               # [1, 2, h, w]
            disparity_map = disparities[0][:, 0, :, :]                 # 1, 1, h, w
            loss_term = loss(disparities, [l_image, r_image])

            if batch in indices:
                samples.append([
                    l_image.squeeze().cpu().numpy(),                   # 3, h, w
                    r_image.squeeze().cpu().numpy(),                   # 3, h, w
                    disparity_map.squeeze(0).cpu().numpy(),            # 1, h, w
                ])

            total_loss += float(loss_term.item()) / float(l_image.shape[0])

        total_loss /= batch + 1

    return total_loss, samples


def show_results(samples):
    
    transform_image = lambda im: im.transpose(1, 2, 0)
    transform_disps = lambda im: im
    transform_depth = lambda im: DisparityToDepth(0.1, 100)(im)

    for sample in samples:
        
        imagL, imagR, disps = sample
        plt.figure(figsize=(18, 8))

        plt.subplot(221)
        plt.imshow(transform_image(imagL))
        plt.title("Left Image (input)")
        plt.axis("off")
        
        plt.subplot(222)
        plt.imshow(transform_image(imagR))
        plt.title("Right Image")
        plt.axis("off")

        plt.subplot(223)
        plt.imshow(transform_disps(disps), cmap="magma", vmax=np.percentile(transform_disps(disps), 95))
        plt.title("Disparities")
        plt.axis("off")
        
        plt.subplot(224)
        plt.imshow(transform_depth(disps), cmap="magma", vmax=np.percentile(transform_depth(disps), 95))
        plt.title("Depth")
        plt.axis("off")

        plt.show()
