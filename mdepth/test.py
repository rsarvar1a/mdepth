
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from .transforms import DisparityToDepth


def test (model, *, dataloader, device, loss, num_samples):
    
    model.eval()
    
    samples = []
    total_loss = 0.
    
    with torch.no_grad():
        
        for batch, data in enumerate(dataloader):
            
            l_image, r_image = data[0].to(device), data[1].to(device) # 1, 3, h, w
            disparities = model(l_image) # [1, 2, h, w]
            disparity_map = disparities[:, 0, :, :] # 1, 1, h, w
            depth_map = DisparityToDepth(1, 100)(disparity_map) # 1, 1, h, w
            loss_term = loss(disparities, [l_image, r_image])
            
            samples.append([
                l_image.squeeze().cpu().numpy(), # 3, h, w
                depth_map.squeeze(0).cpu().numpy() # 1, h, w
            ])
            
            total_loss += np.sum(loss_term.item()) / l_image.shape[0]
    
        total_loss /= (batch + 1)
    
    return total_loss, random.sample(samples, num_samples)


def show_results (samples):
    
    transform_image = lambda im: im.transpose(1, 2, 0)
    transform_depth = lambda im: im
    
    for sample in samples:
        
        image, depth = sample
        
        plt.figure(figsize=(18, 10))
        
        plt.subplot(121)
        plt.imshow(transform_image(image))
        plt.title('Input')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(transform_depth(depth), cmap='magma', vmax=np.percentile(depth, 95))
        plt.title('Depth')
        plt.axis('off')
        
        plt.show()


        