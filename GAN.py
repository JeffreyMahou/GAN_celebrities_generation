import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from parameters import NGPU, LR, BETA1, NZ, NUM_EPOCHS, REAL_LABEL, FAKE_LABEL
from classes import Generator, Discriminator

# custom weights initialization called on ``netG`` and ``netD``

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class GAN():

    def __init__(self):
        
        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
        # Create the generator
        self.generator = Generator(NGPU).to(self.device)
        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (NGPU > 1):
            self.generator = nn.DataParallel(self.generator, list(range(NGPU)))
        
        # Create the Discriminator
        self.discriminator = Discriminator(NGPU).to(self.device)

        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (NGPU > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(NGPU)))

        # Initialize the ``BCELoss`` function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, NZ, 1, 1, device=self.device)

    def initialize_models(self, path_gen=None, path_discr=None):

        if path_gen:
            self.generator.load_state_dict(torch.load(path_gen))
        else:
            # Apply the ``weights_init`` fuNCtion to randomly initialize all weights
            #  to ``mean=0``, ``stdev=0.02``.
            self.generator.apply(weights_init)
        if path_discr:
            self.discriminator.load_state_dict(torch.load(path_discr))
        else:
            # Apply the ``weights_init`` function to randomly initialize all weights
            # like this: ``to mean=0, stdev=0.2``.
            self.discriminator.apply(weights_init)

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=LR, betas=(BETA1, 0.999))

    def show_model(self):

        print("generator : ", self.generator)
        print("discriminator : ", self.discriminator)

    def train_model(self, dataloader):

        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        iters = 0

        avg_errG = 0
        avg_errD = 0
        avg_errD_x = 0
        avg_errD_G_z1 = 0
        avg_errD_G_z2 = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(NUM_EPOCHS):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # Train with all-real batch
                self.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, NZ, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(FAKE_LABEL)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                # (2) Update G network: maximize log(D(G(z)))
                self.generator.zero_grad()
                label.fill_(REAL_LABEL)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                avg_errG += errG
                avg_errD += errD
                avg_errD_x += D_x
                avg_errD_G_z1 += D_G_z1
                avg_errD_G_z2 += D_G_z2

                # Output training stats
                if i % 50 == 0 and i != 0:
                    avg_errG /= 50
                    avg_errD /= 50
                    avg_errD_x /= 50
                    avg_errD_G_z1 /= 50
                    avg_errD_G_z2 /= 50
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, NUM_EPOCHS, i, len(dataloader),
                            avg_errD.item(), avg_errG.item(), avg_errD_x, avg_errD_G_z1, avg_errD_G_z2))
                    
                    avg_errG = 0
                    avg_errD = 0
                    avg_errD_x = 0
                    avg_errD_G_z1 = 0
                    avg_errD_G_z2 = 0

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

    def save_model(self, path_gen, path_discr):

        torch.save(self.discriminator.state_dict(), path_discr)
        torch.save(self.generator.state_dict(), path_gen)

    def plot_losses(self):

        window_size = 20
        G_losses_average = np.convolve(self.G_losses, [1/window_size for i in range(window_size)], mode='valid')
        D_losses_average = np.convolve(self.D_losses, [1/window_size for i in range(window_size)], mode='valid')
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.plot(G_losses_average,label="G_average")
        plt.plot(D_losses_average,label="D_average")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_generator_performance(self):

        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())
        plt.show()

    def test_generator(self, dataloader):

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))

        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1],(1,2,0)))
        plt.show()