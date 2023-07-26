global WORKERS, BATCH_SIZE, IMAGE_SIZE, NC, NZ, NGF, NDF, NUM_EPOCHS, LR, BETA1, NGPU, REAL_LABEL, FAKE_LABEL

# Number of workers for dataloader
WORKERS = 2

# Batch size during training
BATCH_SIZE = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
IMAGE_SIZE = 64

# Number of channels in the training images. For color images this is 3
NC = 3

# Size of z latent vector (i.e. size of generator input)
NZ = 100

# Size of feature maps in generator
NGF = 64

# Size of feature maps in discriminator
NDF = 64

# Number of training epochs
NUM_EPOCHS = 1

# Learning rate for optimizers
LR = 0.0002

# Beta1 hyperparameter for Adam optimizers
BETA1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
NGPU = 0

# Establish convention for real and fake labels during training
REAL_LABEL = 1.
FAKE_LABEL = 0.