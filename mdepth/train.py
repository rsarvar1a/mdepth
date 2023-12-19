import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm


def train(
    model, *, dataloader, device, epochs, loss, optimizer, silent=False
) -> list[float]:
    model.train(mode=True)
    losses = []

    for epoch in range(epochs):
        # Some performance metrics.
        time_s = time.perf_counter()

        total_loss = 0.0
        ap = 0.0
        ds = 0.0
        lr = 0.0

        for batch, data in tqdm.tqdm(
            enumerate(dataloader), unit="batch", total=len(dataloader)
        ):
            optimizer.zero_grad()

            # Each training datapoint is a stereo pair. We train on the left images
            # as the reference frames, and calculate the loss using the right images.
            l_images, r_images = data[0].to(device), data[1].to(device)

            # Compute the disparity maps with the left images as the reference frame,
            # then provide them to the loss function.
            disparity_maps = model(l_images)
            loss_term = loss(disparity_maps, [l_images, r_images])

            # Update the loss by adding the average loss of the batch.
            s = float(l_images.shape[0])
            total_loss += float(loss_term.item()) / s
            ap += float(loss.loss_ap) / s
            ds += float(loss.loss_ds) / s
            lr += float(loss.loss_lr) / s

            # Train the model.
            loss_term.backward()
            optimizer.step()

        # The loss is the average loss per batch over all batches, rather than the sum
        # over all batches.
        total_loss /= batch + 1
        ap /= batch + 1
        ds /= batch + 1
        lr /= batch + 1

        losses.append((total_loss, ap, ds, lr))

        time_e = time.perf_counter()

        if not silent:
            print(
                "Epoch {epoch}: {seconds} sec, {loss} loss\n".format(
                    epoch=str(epoch + 1).zfill(3),
                    seconds=str(round(time_e - time_s)).zfill(4),
                    loss=str(round(total_loss, 4)).ljust(6, '0'),
                )
            )

    model.train(mode=False)
    return losses


def display_loss_graph(losses):
    losses = zip(*losses)
    main, ap, ds, lr = (np.array(l) for l in losses)

    plt.figure(figsize=(12, 9))

    ax = plt.subplot(211)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("proportional loss")
    ax.plot(main / main[0], label="total loss")
    ax.plot(ap / ap[0], label="image loss")
    ax.plot(ds / ds[0], label="disparity smoothness loss")
    ax.plot(lr / lr[0], label="LR-consistency loss")
    ax.legend(loc="upper right")

    ax = plt.subplot(212)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("absolute loss")
    ax.plot(main, label="total loss")
    ax.plot(ap, label="image loss")
    ax.plot(ds, label="disparity smoothness loss")
    ax.plot(lr, label="LR-consistency loss")
    ax.legend(loc="upper right")

    plt.show()
