
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from .transforms import DisparityToDepth


def test (model, *, dataloader, device, num_samples):
    
    model.eval()
    
    samples = []
    
    with torch.no_grad():
        
        for batch, data in enumerate(dataloader):
            
            inputs = data.to(device) # 1, 3, h, w
            disparities = model(inputs) # [1, 2, h, w]
            disparity_map = disparities[0][:, 0, :, :] # 1, 1, h, w
            depth_map = DisparityToDepth(1, 100)(disparity_map) # 1, 1, h, w
                        
            samples.append([
                inputs.squeeze().cpu().numpy(), # 3, h, w
                depth_map.squeeze(0).cpu().numpy() # 1, h, w
            ])
    
    return random.sample(samples, num_samples)


def show_results (samples):
    
    transform = lambda im: im.permute(1, 2, 0)
    
    for sample in samples:
        
        image, depth = sample
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        plt.imshow(transform(image))
        plt.title('Input')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(transform(depth), cmap='magma', vmax=np.percentile(depth, 95))
        plt.title('Depth')
        plt.axis('off')
        
        plt.show()


        