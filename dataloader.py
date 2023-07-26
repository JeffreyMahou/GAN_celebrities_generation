import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from parameters import IMAGE_SIZE, BATCH_SIZE, WORKERS, NGPU

global dataloader, device

class DataLoader():

    def __init__(self, dataroot):

        self.dataroot = dataroot

    def load_data(self):

        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        dataset = dset.ImageFolder(root=self.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(IMAGE_SIZE),
                                    transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=WORKERS)

    def plot_sample(self):

        # Plot some training images
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()