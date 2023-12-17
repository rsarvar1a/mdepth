
import matplotlib.pyplot as plt
import numpy as np
import time


def train (model, *, dataloader, device, epochs, loss, optimizer, silent = False) -> list[float]:
    
    model.train(mode=True)
    losses = []
    
    for epoch in range(epochs):

        # Some performance metrics.
        time_s = time.perf_counter()
        
        total_loss = 0.
        
        for batch, data in enumerate(dataloader):
            
            optimizer.zero_grad()
            
            # Each training datapoint is a stereo pair. We train on the left images
            # as the reference frames, and calculate the loss using the right images.
            l_images, r_images = data[0].to(device), data[1].to(device)
            
            # Compute the disparity maps with the left images as the reference frame,
            # then provide them to the loss function.
            disparity_maps = model(l_images)
            loss_term = loss(disparity_maps, [l_images, r_images])
            
            # Train the model.
            loss_term.backward()
            optimizer.step()
            
            # Update the loss by adding the average loss of the batch.
            total_loss += np.sum(loss_term.item()) / l_images.shape[0]
        
        # The loss is the average loss per batch over all batches, rather than the sum 
        # over all batches.
        total_loss /= (batch + 1)
        losses.append(total_loss)
        
        time_e = time.perf_counter()
        
        if not silent:
            print(
                "Epoch {epoch}: {seconds} sec, {loss} loss".format(
                    epoch=str(epoch + 1).rjust(3),
                    seconds=str(round(time_e - time_s)).rjust(4),
                    loss=str(round(loss, 4)).rjust(5)
                )
            )
    
    model.train(mode=False)
    return losses


def display_loss_graph (losses):
    
    plt.figure(figsize=(12, 8))
    
    ax = plt.add_subplot(111)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('training loss')
    ax.plot(losses)
    
    plt.show()