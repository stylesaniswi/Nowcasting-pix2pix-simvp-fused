class Options:
    def __init__(self):
        self.dataset_mode = 'aligned'
        self.input_nc = 3  # Assuming RGB images as input
        self.output_nc = 1  # Assuming RGB images as output
        self.epoch = 'latest'

        self.netG = 'unet_128'
        self.norm = 'batch'
        self.ngf = 64
        self.ndf=64

        self.isTrain = False  # Set to True during training
        self.gan_mode = 'vanilla'
        self.lambda_L1 = 100.0
        self.lr = 0.0002
        self.beta1 = 0.5
        self.n_epochs = 100
        self.n_epochs_decay = 100

        self.gpu_ids = []  # Use GPU if available, e.g., [0] for GPU 0
        self.batch_size = 2
        self.pool_size = 0  # No image buffer during inference
        self.init_type = 'normal'
        self.init_gain = 0.02

# Create an instance of Options and set your desired options
def get_opt_pix2pix():
    opt = Options()
    return opt
