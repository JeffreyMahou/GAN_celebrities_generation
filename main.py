
import argparse
import random
import os
import torch

from dataloader import DataLoader
from GAN import GAN

def main():

    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    torch.autograd.set_detect_anomaly(True)

    # Root directory for dataset
    parser = argparse.ArgumentParser(description='Computing ')
    parser.add_argument('--path', type=str, help='path to the folder', required=True) # path to the image file
    parser.add_argument('--name_images', type=str, help='name of the images folder', required=True) # path to the image file
    parser.add_argument('--name_gen', type=str, help='name of the generator weights file', default=None, required=False) # path to the image file
    parser.add_argument('--name_discr', type=str, help='name of the discriminator weights file', default=None, required=False) # path to the image file
    parser.add_argument('--save', type=int, help='whether or not to save the weights', default=0, required=False)
    args = parser.parse_args()
    dataroot = args.path
    name_images = args.name_images
    path_gen = os.path.join(dataroot, args.name_gen) if args.name_gen else None
    path_discr = os.path.join(dataroot, args.name_discr) if args.name_discr else None
    save = bool(args.save)

    DL = DataLoader(os.path.join(dataroot, name_images))
    DL.load_data()
    DL.plot_sample()

    G = GAN()
    G.initialize_models(path_gen, path_discr)
    G.show_model()
    G.train_model(DL.dataloader)
    if save:
        G.save_model(path_gen, path_discr)
    G.plot_losses()
    G.plot_generator_performance()
    G.test_generator(DL.dataloader)

if __name__ == '__main__':
    main()