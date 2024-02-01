import csv
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import math

class ImageQualityMetrics:
    """Class to compute image quality metrics like SSIM, PSNR, and MSE."""

    @staticmethod
    @staticmethod
    def ssim_3d(img1, img2):
        """Calculate SSIM for each 2D slice in the 3D image and return the average."""
        ssim_vals = []
        for i in range(img1.shape[1]):  # Depth
            slice1 = img1[0, i, :, :]
            slice2 = img2[0, i, :, :]
            ssim_val = compare_ssim(slice1, slice2, data_range=slice1.max() - slice1.min())
            ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    
    @staticmethod
    def psnr(img1, img2):
        """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images."""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return math.inf
        return 20 * math.log10(img1.max() - img1.min()) - 10 * math.log10(mse)

    @staticmethod
    def mse(img1, img2):
        """Calculate MSE (Mean Squared Error) between two images."""
        return torch.mean((img1 - img2) ** 2)


class ValidationRecorder:
    """Class to handle validation process and record the metrics."""

    def __init__(self, csv_file_path):
        """Initialize the recorder with the path to the CSV file."""
        self.csv_file_path = csv_file_path

    def initialize_csv(self):
        """Initialize the CSV file with headers."""
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Loss', 'SSIM', 'PSNR', 'MSE'])

    def validate_and_record(self, epoch, dataloader, device, generator, 
                            criterion_g):
        """Validate the model and record the metrics in the CSV file."""
        generator.eval()
        total_loss, total_ssim, total_psnr, total_mse = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for _, (low_res, high_res) in enumerate(dataloader):
                low_res, high_res = low_res.to(device), high_res.to(device)
                fake_images = generator(low_res)

                loss = criterion_g(fake_images, high_res)
                total_loss += loss.item()

                for j in range(high_res.size(0)):
                    ssim_val = ImageQualityMetrics.ssim_3d(
                        high_res[j].cpu().numpy(), fake_images[j].cpu().numpy())
                    psnr_val = ImageQualityMetrics.psnr(
                        high_res[j], fake_images[j])
                    mse_val = ImageQualityMetrics.mse(
                        high_res[j], fake_images[j])
                    total_ssim += ssim_val
                    total_psnr += psnr_val
                    total_mse += mse_val.item()

        avg_loss = total_loss / len(dataloader)
        avg_ssim = total_ssim / (len(dataloader) * dataloader.batch_size)
        avg_psnr = total_psnr / (len(dataloader) * dataloader.batch_size)
        avg_mse = total_mse / (len(dataloader) * dataloader.batch_size)

        self._write_to_csv(epoch, avg_loss, avg_ssim, avg_psnr, avg_mse)

    def _write_to_csv(self, epoch, avg_loss, avg_ssim, avg_psnr, avg_mse):
        """Write the validation metrics to the CSV file."""
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_loss, avg_ssim, avg_psnr, avg_mse])
