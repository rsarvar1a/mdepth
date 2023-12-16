
import datetime
import torch

from torch.utils.data import DataLoader

from .mdepth.data import KittiDataset
from .mdepth.model import MonocularDepth
from .mdepth.train import train, display_loss_graph
from .mdepth.transforms.preprocess import JointToTensor, JointRandomResizeCrop, JointNormalize
from .mdepth.utils.loss import MonocularLoss


def main():
    
    transforms = [
        JointToTensor(),
        JointRandomResizeCrop(400, 0.8, 1.25),
        JointNormalize([0.485, 0.456, 0.406], [0.225, 0.224, 0.225])
    ]
    
    dataset = KittiDataset("data/Kitti2015", mode=train, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    device = torch.device('cuda:0')
    net = MonocularDepth().to(device)
    optimizer = torch.Adam(net.parameters(), lr=2e-5)
    
    losses = train(
        net, 
        device=device, 
        dataloader=dataloader,
        epochs=10,
        loss=MonocularLoss(net.decoder.length),
        optimizer=optimizer
    )
    display_loss_graph(losses)
    
    net.save(f"data/pretrained/{datetime.now()}.pt")

if __name__ == "__main__":
    main()