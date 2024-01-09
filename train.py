import os
import glob
import argparse
import torch
import torch.nn as nn
import torchio as tio
import random
import numpy as np
import nibabel as nib
import logging
from scipy.ndimage import zoom
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from network.generator import ResnetGenerator
from network.discriminator import PatchDiscriminator
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt


def check_cuda():
    """Checks CUDA availability and prints GPU information."""
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Availability: {cuda_available}")
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Available number of GPU(s): {num_gpus}")
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_info.name}, Memory: {gpu_info.total_memory / 1e9}GB")


def setup_logging(log_file):
    """Sets up logging configuration."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class MRIDataset(Dataset):
    """
    A custom Dataset class for MRI images.
    It loads high resolution images, resamples them to lower resolutions,
    normalizes and rotates them for the training process.
    """

    def __init__(self, file_list, low_res_shape, high_res_shape, augment=False):
        self.file_list = file_list
        self.low_res_shape = low_res_shape
        self.high_res_shape = high_res_shape
        self.augment = augment
        self.epoch = 0
        self.update_transforms()

    def __len__(self):
        return len(self.file_list)
    
    def update_transforms(self):
        if self.augment:
            augmentation_methods = [
                tio.RandomAffine(degrees=15, translate=(0.1, 0.1, 0.1), scale=(0.9, 1.1)),
                tio.RandomFlip(axes=(0,)),
            ]
            selected_transforms = random.sample(augmentation_methods, 
                                                k=random.randint(1, len(augmentation_methods)))
            self.transforms = tio.Compose(selected_transforms)
        else:
            self.transforms = None

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        high_res_image = nib.load(file_path).get_fdata()
        low_res_image = self.resample_image(high_res_image, self.low_res_shape)

        high_res_image = self.normalize_image(high_res_image)
        low_res_image = self.normalize_image(low_res_image)

        high_res_tensor = torch.tensor(np.expand_dims(high_res_image, axis=0), dtype=torch.float16)
        low_res_tensor = torch.tensor(np.expand_dims(low_res_image, axis=0), dtype=torch.float16)

        if self.transforms:
            subject = tio.Subject(high_res=tio.ScalarImage(tensor=high_res_tensor))
            subject = self.transforms(subject)
            high_res_tensor = subject['high_res'].data

        return low_res_tensor, high_res_tensor

    def rotate_image(self, image):
        return np.rot90(image, k=1, axes=(1, 2))

    def normalize_image(self, image):
        image = (image - np.mean(image)) / np.std(image)
        return 255 * (image - np.min(image)) / (np.max(image) - np.min(image))

    def resample_image(self, image, target_shape):
        factors = (
            target_shape[0] / image.shape[0],
            target_shape[1] / image.shape[1],
            target_shape[2] / image.shape[2]
        )
        return zoom(image, factors, order=1)
    
    def on_epoch_end(self):
        self.epoch += 1
        self.update_transforms()


def train(epoch, epochs, dataloader, device, generator, discriminator, criterion_g, criterion_d,
          optimizer_g, optimizer_d, scaler, best_loss, ckpt_final_path, log_file, save_dir, checkpoint_freq):
    """
    The training function for each epoch.
    It iterates over the dataloader, computes the loss for generator and discriminator,
    and saves checkpoints at specified intervals.
    """
    global_lowest_loss = best_loss
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
    total_loss_g = 0.0
    total_loss_d = 0.0
    for i, (low_res, high_res) in loop:
        low_res, high_res = low_res.to(device), high_res.to(device)
        real_labels = torch.ones(high_res.size(0), 1, device=device)
        fake_labels = torch.zeros(high_res.size(0), 1, device=device)

        # Generator update
        optimizer_g.zero_grad()
        with autocast():
            fake_images = generator(low_res)
            loss_g = criterion_g(fake_images, high_res)
        scaler.scale(loss_g).backward()
        scaler.step(optimizer_g)
        scaler.update()

        # Discriminator update
        optimizer_d.zero_grad()
        with autocast():
            loss_real = criterion_d(discriminator(high_res), real_labels)
            loss_fake = criterion_d(discriminator(fake_images.detach()), fake_labels)
            loss_d = (loss_real + loss_fake) / 2
        scaler.scale(loss_d).backward()
        scaler.step(optimizer_d)
        scaler.update()

        total_loss_g += loss_g.item()
        total_loss_d += loss_d.item()
        avg_loss_g = round(total_loss_g / (i + 1), 4)
        avg_loss_d = round(total_loss_d / (i + 1), 4)
        loop.set_postfix(loss_g=avg_loss_g, loss_d=avg_loss_d)

        dataloader.dataset.on_epoch_end()

    # Save checkpoints
    if (epoch + 1) % checkpoint_freq == 0:
        torch.save(generator.state_dict(), os.path.join(save_dir, f'generator_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, f'discriminator_{epoch+1}.pth'))
        logging.info(f"Checkpoint saved for epoch {epoch+1} in {save_dir}")

    # Save best checkpoint
    if avg_loss_g < global_lowest_loss:
        global_lowest_loss = avg_loss_g
        torch.save(generator.state_dict(), os.path.join(ckpt_final_path, 'G_latest.pth'))
        torch.save(discriminator.state_dict(), os.path.join(ckpt_final_path, 'D_latest.pth'))
        logging.info(f"New best checkpoint saved in {ckpt_final_path} with loss {avg_loss_g}")

    return global_lowest_loss


def load_rotate_and_save_first_image(dataloader, output_path, high_res_shape, low_res_shape):
    """
    Loads the first batch from the dataloader, extracts and saves the middle slices
    of the high and low resolution images in the specified output path.
    """
    first_data = next(iter(dataloader))
    _, high_res = first_data
    high_res_image = high_res.squeeze().numpy()
    low_res_image = zoom(high_res_image, (low_res_shape[0] / high_res_image.shape[0], 
                                          low_res_shape[1] / high_res_image.shape[1], 
                                          low_res_shape[2] / high_res_image.shape[2]), 
                                          order=1)
    high_res_rotated = np.rot90(high_res_image, k=1, axes=(1, 2))
    low_res_rotated = np.rot90(low_res_image, k=1, axes=(1, 2))
    
    save_middle_slice(high_res_rotated, os.path.join(output_path, 'high_res_first_slice.png'))
    save_middle_slice(low_res_rotated, os.path.join(output_path, 'low_res_first_slice.png'))


def save_middle_slice(image, file_path):
    """
    Saves the middle slice of a given 3D image to the specified file path.
    """
    middle_slice = image[image.shape[0] // 2]
    plt.imshow(middle_slice, cmap='gray')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    """
    Main function to initialize the GAN model, set up the dataset and dataloader,
    and execute the training process.
    """
    check_cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='ckpt')
    parser.add_argument('--final', nargs='?', const='ckpt/ckpt_final', default=None, 
                        help="Continue training from final checkpoint. Optionally specify the checkpoint folder.")
    parser.add_argument('--low_res_shape', nargs=3, type=int, default=[128, 32, 128])
    parser.add_argument('--high_res_shape', nargs=3, type=int, default=[128, 192, 128])
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to a previous checkpoint")
    parser.add_argument('--checkpoint_freq', type=int, default=50, help="Frequency of saving checkpoints (in epochs)")
    args = parser.parse_args()

    # Setup device and directories
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)
    setup_logging(os.path.join(args.save_path, 'training_log.txt'))
    logging.info("Training started")

    # Prepare data
    file_list = glob.glob('train_data/*.nii') + glob.glob('train_data/*.nii.gz')
    if not file_list:
                raise RuntimeError("No training data found in 'train_data' directory.")
    dataloader = DataLoader(MRIDataset(file_list, args.low_res_shape, args.high_res_shape),
                            batch_size=args.batch_size, shuffle=True)

    # Initialize the GAN components
    generator = ResnetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    criterion_g = nn.MSELoss()
    criterion_d = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load checkpoint if available
    start_epoch, best_loss = 0, float('inf')
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss_g']
        print(f"Loaded checkpoint from epoch {start_epoch}")
    
    # Check if 'final' flag is used to set the checkpoint folder path
    if args.final:
        checkpoint_folder = args.final

        # Find generator and discriminator checkpoint files in the specified folder
        gen_ckpt = glob.glob(os.path.join(checkpoint_folder, 'g*.pth')) + \
                   glob.glob(os.path.join(checkpoint_folder, 'G*.pth'))
        disc_ckpt = glob.glob(os.path.join(checkpoint_folder, 'd*.pth')) + \
                    glob.glob(os.path.join(checkpoint_folder, 'D*.pth'))

        # Load the checkpoints if they exist
        if gen_ckpt and disc_ckpt:
            generator.load_state_dict(torch.load(gen_ckpt[0]))
            discriminator.load_state_dict(torch.load(disc_ckpt[0]))
            print(f"Loaded generator and discriminator checkpoints "
                  f"from {checkpoint_folder}")
        else:
            print("No generator or discriminator checkpoint found "
                  "in specified folder")

    # Prepare directories for saving outputs
    existing_dirs = glob.glob(os.path.join(args.save_path, 'easysr_*'))
    next_folder_number = len(existing_dirs) + 1
    save_dir = os.path.join(args.save_path, f'easysr_{next_folder_number:03d}')
    os.makedirs(save_dir)
    ckpt_final_path = os.path.join(args.save_path, 'ckpt_final')
    os.makedirs(ckpt_final_path, exist_ok=True)
    log_file = os.path.join(ckpt_final_path, 'best_checkpoints_log.txt')

    # Training process
    scaler = GradScaler()
    for epoch in range(start_epoch, args.epochs):
        best_loss = train(epoch, args.epochs, dataloader, device, generator, 
                          discriminator, criterion_g, criterion_d, optimizer_g, 
                          optimizer_d, scaler, best_loss, ckpt_final_path, log_file, 
                          save_dir, args.checkpoint_freq)
        logging.info(f"Epoch {epoch+1}/{args.epochs} completed")

    logging.info("Training completed")

if __name__ == "__main__":
    main()

