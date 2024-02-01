import argparse
import os
import glob
import numpy as np
import nibabel as nib
import torch
import time
from network.generator import ResnetGenerator
from scipy.ndimage import zoom
from tqdm import tqdm
import ants
import subprocess
from pathlib import Path
import tempfile
import shutil

# Class to handle MRI image inference
class MRIInference:
    def __init__(self, model, device, input_shape, output_shape):
        # Initialize with model, device, and shapes
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.output_shape = output_shape

    def load_image(self, file_path, intensity_adjust):
        # Load the image using nibabel
        nib_image = nib.load(file_path)
        image_data = nib_image.get_fdata()
                
        rotated_image = np.rot90(image_data, k=1, axes=(1, 2))
        
        if intensity_adjust:
            # Min-Max normalization to 0-783
            min_val, max_val = np.min(rotated_image), np.max(rotated_image)
            scale = 783 / (max_val - min_val)
            normalized_image = scale * (rotated_image - min_val)

            # Saturate values above 255
            normalized_image = np.clip(normalized_image, 0, 255)
        else:
            # Standard normalization to 0-255
            mean, std = np.mean(rotated_image), np.std(rotated_image)
            normalized_image = (rotated_image - mean) / std
            min_val, max_val = np.min(normalized_image), np.max(normalized_image)
            scale = 255 / (max_val - min_val)
            normalized_image = scale * (normalized_image - min_val)

        scale_factors = (
            self.input_shape[0] / normalized_image.shape[0], 
            self.input_shape[1] / normalized_image.shape[1], 
            self.input_shape[2] / normalized_image.shape[2]
        )
        resampled_image = zoom(normalized_image, scale_factors, order=3)
        return torch.tensor(resampled_image[np.newaxis, np.newaxis, ...], dtype=torch.float32)

    def save_image(self, image, file_name):
        # Save processed image to file
        image = image.squeeze().cpu().numpy()
        scale_factors = (
            self.output_shape[0] / image.shape[0], 
            self.output_shape[1] / image.shape[1], 
            self.output_shape[2] / image.shape[2]
        )
        resampled_image = zoom(image, scale_factors, order=3)
        #resampled_image = np.rot90(resampled_image, k=-1, axes=(1, 2))
        nib.save(nib.Nifti1Image(resampled_image, np.eye(4)), file_name)

    def match_sform_affine(self, orig_path, gen_path):
        # Match affine transformation of original and generated images
        orig_img = nib.load(orig_path)
        orig_affine = orig_img.affine
        gen_img = nib.load(gen_path)
        gen_data = gen_img.get_fdata()
        matched_gen_img = nib.Nifti1Image(gen_data, orig_affine)
        nib.save(matched_gen_img, gen_path)

    def infer(self, aligned_image_path, original_file_path, output_path, enhance):
        # Load and preprocess the image from aligned_image_path
        input_tensor = self.load_image(aligned_image_path, enhance)
        
        # Perform inference on input tensor
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor.to(self.device))
        
        # Resample output to target shape
        scale_factor = (
            self.output_shape[0] / output.shape[2],
            self.output_shape[1] / output.shape[3],
            self.output_shape[2] / output.shape[4]
        )
        resampled_output = zoom(
            output.squeeze().cpu().numpy(), scale_factor, order=3)
        generated_image = torch.tensor(resampled_output[np.newaxis, ...])
        
        # Save the generated image
        temp_generated_path = os.path.join(output_path, 'temp_generated.nii.gz')
        self.save_image(generated_image, temp_generated_path)

        # Get and print orientation code of the original image
        orig_img = ants.image_read(original_file_path)
        orig_orientation = ants.get_orientation(orig_img)
        #print(f"Orientation of the original image: {orig_orientation}")  ## print Orientation ##

        # Reorient the generated image based on original orientation
        gen_img = nib.load(temp_generated_path)
        gen_data = gen_img.get_fdata()
        reoriented_image = ants.from_numpy(gen_data)

        if orig_orientation == 'LSP':
            reoriented_image = ants.reorient_image2(reoriented_image, 'RAI')
        elif orig_orientation == 'LPI':
            reoriented_image = ants.reorient_image2(reoriented_image, 'RIP')
        elif orig_orientation == 'RAS':
            reoriented_image = ants.reorient_image2(reoriented_image, 'LSA')
        # No reorientation for other cases

        # Save the reoriented image
        nib.save(nib.Nifti1Image(reoriented_image.numpy(), np.eye(4)), temp_generated_path)

        # Match affine and resample
        temp_orig_path = os.path.join(output_path, 'temp_orig.nii.gz')
        resampled_file_path = resample_to_isotropic(
            original_file_path, temp_orig_path)
        self.match_sform_affine(resampled_file_path, temp_generated_path)
        
        resampled_generated_path = os.path.join(output_path, 'resampled_generated.nii.gz')
        resample_to_isotropic(temp_generated_path, resampled_generated_path)

        base_name = os.path.basename(original_file_path)
        gen_file_name = f"{Path(base_name).stem}_{int(time.time())}_gen.nii.gz"
        warped_file_path = os.path.join(output_path, gen_file_name)
        affine_registration(
            resampled_file_path, temp_generated_path, warped_file_path)

        # Remove temporary files
        for temp_file in [temp_orig_path, temp_generated_path, resampled_generated_path]:
            os.remove(temp_file)
        
        return warped_file_path
    
    # Perform inference and handle images
    def run_inference(self, input_tensor, temp_file_path, output_path):
        try:
            warped_image_path = self.infer(input_tensor, temp_file_path, output_path)

            gen_file_name = temp_file_path.replace(".nii", "_gen.nii")
            download_file_path = os.path.join(output_path, gen_file_name)
            shutil.copy(warped_image_path, download_file_path)

            return (download_file_path, gen_file_name)
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None

# Image processing functions
def resample_to_isotropic(image_path, output_path):
    # Resample image to isotropic resolution
    image = ants.image_read(image_path)
    resampled_image = ants.resample_image(
        image, (0.15, 0.15, 0.15), use_voxels=False, interp_type=3)
    ants.image_write(resampled_image, output_path)
    return output_path

def affine_registration(fixed_image_path, moving_image_path, output_path):
    # Perform affine registration between two images
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    registration = ants.registration(
        fixed=fixed_image, moving=moving_image, 
        type_of_transform='Rigid')
    ants.image_write(registration['warpedmovout'], output_path)

def align_to_template(resampled_image_path, template_path, output_path):
    # Align the resampled image to the template
    moving_image = ants.image_read(resampled_image_path)
    fixed_image = ants.image_read(template_path)
    registration = ants.registration(
        fixed=fixed_image, moving=moving_image, 
        type_of_transform='Rigid')
    aligned_image = registration['warpedmovout']
    ants.image_write(aligned_image, output_path)
    return output_path

def clear_output_folder(folder_path):
    # Clear contents of a specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def download_model_if_needed(templates_folder):
    """Downloads model from Hugging Face if template folder is empty or doesn't exist."""
    if not os.path.exists(templates_folder) or not os.listdir(templates_folder):
        print("Downloading model from Hugging Face...")
        os.makedirs(templates_folder, exist_ok=True)
        subprocess.run(["huggingface-cli", "download", "hwonheo/easysr_templates", 
                        "--local-dir", templates_folder, "--local-dir-use-symlinks", "False"], check=True)

def load_model(model_choice):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = ResnetGenerator().to(device)

    if model_choice == "T1-Model":
        checkpoint_path = 'ckpt/ckpt_final/G_latest_T1.pth'
    else: # "Mixed-Model"
        checkpoint_path = 'ckpt/ckpt_final/G_latest_Mixed.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    return generator, device

def validate_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Specified path does not exist: {path}")

def main():
    parser = argparse.ArgumentParser(
                        description="EasySR: Easy Web UI for Generative 3D Inference of Rat Brain MRI")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the input MRI image(s).")
    parser.add_argument('--output', type=str, default='infer/generate/',
                        help="Path to the output folder. Default: infer/generate/")
    parser.add_argument('--model_choice', type=str, default='Mixed-Model',
                        choices=['Mixed-Model', 'T1-Model'],
                        help="Choose the model type: Combined-Model, T1-Model, T2-Model. Default: "
                        "Pre-trained-T1_T2-Combined-Model")
    parser.add_argument('--templates_folder', type=str, default='templates',
                        help="Path to the folder containing templates. Default: templates")
    parser.add_argument('--enhance', action='store_true',
                    help="Enable signal intensity enhancement for low signal intensity MRI images. "
                         "This can improve the clarity and definition of the images.")

    args = parser.parse_args()

    # Validate input and output path
    validate_path(args.input)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Validate template file
    templates_folder = args.templates_folder
    validate_path(templates_folder)
    template_path = os.path.join(templates_folder, "bmc_t2_rat.nii.gz")
    validate_path(template_path)

    # Ensure template files are available
    download_model_if_needed(args.templates_folder)
    template_path = os.path.join(args.templates_folder, "bmc_t2_rat.nii.gz")  # Use the T2 template

    # Initialize model and inference engine
    generator, device = load_model(args.model_choice)
    inference_engine = MRIInference(generator, device, (128, 128, 64), (128, 128, 192))

    # Input path can be a single file or a directory
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = glob.glob(os.path.join(args.input, '*.nii*'))

    for file in tqdm(files, desc="Processing MRI Images"):
        # Temporary directory for file processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, os.path.basename(file))

            # Copy file to temporary directory
            shutil.copy(file, temp_file_path)

            # Resample and align the image
            resampled_path = resample_to_isotropic(
                temp_file_path, os.path.join(temp_dir, "resampled.nii.gz"))
            aligned_path = align_to_template(
                resampled_path, template_path, os.path.join(temp_dir, "aligned.nii.gz"))

            try:
                # Perform inference using the original and aligned image
                warped_image_path = inference_engine.infer(
                aligned_path, temp_file_path, args.output, args.enhance)

                gen_file_name = os.path.basename(file).replace(".nii", "_gen.nii")
                download_file_path = os.path.join(args.output, gen_file_name)
                shutil.copy(warped_image_path, download_file_path)

                print(f"Processed image saved at: {download_file_path}")
            except Exception as e:
                print(f"Error during processing: {str(e)}")

        print("All processing completed.")

if __name__ == '__main__':
    main()