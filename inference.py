import argparse
import os
import glob
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from tqdm import tqdm
from network.generator import ResnetGenerator
import ants

def resample_to_isotropic(image_path, output_path):
    image = ants.image_read(image_path)
    resampled_image = ants.resample_image(
        image, (0.15, 0.15, 0.15), use_voxels=False, interp_type=4)
    ants.image_write(resampled_image, output_path)
    return output_path

def affine_registration(fixed_image_path, moving_image_path, output_path):
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    registration = ants.registration(
        fixed=fixed_image, moving=moving_image, 
        type_of_transform='Elastic')
    ants.image_write(registration['warpedmovout'], output_path)

class MRIInference:
    def __init__(self, model, device, input_shape, output_shape):
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.output_shape = output_shape

    def load_image(self, file_path):
        image = nib.load(file_path).get_fdata()
        rotated_image = np.rot90(image, k=1, axes=(1, 2))

        mean = np.mean(rotated_image)
        std = np.std(rotated_image)
        normalized_image = (rotated_image - mean) / std

        min_val = np.min(normalized_image)
        max_val = np.max(normalized_image)
        scale = 255 / (max_val - min_val)
        normalized_image = scale * (normalized_image - min_val)

        scale_factors = (
            self.input_shape[0] / normalized_image.shape[0], 
            self.input_shape[1] / normalized_image.shape[1], 
            self.input_shape[2] / normalized_image.shape[2]
        )
        resampled_image = zoom(normalized_image, scale_factors, order=1)
        return torch.tensor(
            resampled_image[np.newaxis, np.newaxis, ...], dtype=torch.float32)

    def save_image(self, image, file_name):
        image = image.squeeze().cpu().numpy()
        scale_factors = (
            self.output_shape[0] / image.shape[0], 
            self.output_shape[1] / image.shape[1], 
            self.output_shape[2] / image.shape[2]
        )
        resampled_image = zoom(image, scale_factors, order=1)
        resampled_image = np.rot90(resampled_image, k=-1, axes=(1, 2))
        nib.save(nib.Nifti1Image(resampled_image, np.eye(4)), file_name)

    def infer(self, input_tensor, original_file_path, output_path):
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor.to(self.device))

        scale_factor = (1, 1, self.output_shape[2] / output.shape[4])
        resampled_output = zoom(
            output.squeeze().cpu().numpy(), scale_factor, order=1)
        generated_image = torch.tensor(resampled_output[np.newaxis, ...])

        temp_orig_path = os.path.join(output_path, 'temp_orig.nii.gz')
        resampled_file_path = resample_to_isotropic(
            original_file_path, temp_orig_path)

        temp_generated_path = os.path.join(output_path, 'temp_generated.nii.gz')
        self.save_image(generated_image, temp_generated_path)

        self.match_sform_affine(resampled_file_path, temp_generated_path)

        resampled_generated_path = os.path.join(output_path, 'resampled_generated.nii.gz')
        resample_to_isotropic(temp_generated_path, resampled_generated_path)

        base_name = os.path.basename(original_file_path)
        gen_file_name = base_name.replace(".nii", "_gen.nii")
        warped_file_path = os.path.join(output_path, gen_file_name)
        affine_registration(
            resampled_file_path, resampled_generated_path, warped_file_path)

        for temp_file in [temp_orig_path, temp_generated_path, resampled_generated_path]:
            os.remove(temp_file)

        return warped_file_path

    def match_sform_affine(self, orig_path, gen_path):
        orig_img = nib.load(orig_path)
        orig_affine = orig_img.affine

        gen_img = nib.load(gen_path)
        gen_data = gen_img.get_fdata()

        matched_gen_img = nib.Nifti1Image(gen_data, orig_affine)
        nib.save(matched_gen_img, gen_path)

def main():
    parser = argparse.ArgumentParser(description="EasySR: Rat Brain T2-MRI SR Inference")
    parser.add_argument('--input', type=str, default=None,
                        help="Path to the input MRI image(s). Default: infer/input/")
    parser.add_argument('--ckpt', type=str, default='ckpt/ckpt_final/G_latest.pth',
                        help="Path to the checkpoint file. Default: ckpt/ckpt_final/G_latest.pth")
    parser.add_argument('--output', type=str, default='infer/generate/',
                        help="Path to the output folder. Default: infer/generate/")

    args = parser.parse_args()

    input_path = args.input if args.input else 'infer/input/'
    checkpoint_path = args.ckpt
    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = ResnetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint)

    inference_engine = MRIInference(generator, device, (128, 32, 128), (128, 192, 128))

    # Input path can be a single file or a directory
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = glob.glob(os.path.join(input_path, '*.nii*'))

    for file in tqdm(files, desc="Inferring MRI Images"):
        input_tensor = inference_engine.load_image(file)
        warped_image_path = inference_engine.infer(
            input_tensor, file, output_path)
        print(f"Warped image saved at: {warped_image_path}")

    print("Inference completed.")

if __name__ == '__main__':
    main()