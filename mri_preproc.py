import os
import sys
import argparse
import ants
from tqdm import tqdm
import logging
import subprocess

class MRIProcessor:
    """Class for processing MRI images using AntsPy for affine registration."""
    
    def __init__(self, input_path, output_folder, fixed_image_path):
        self.input_path = input_path
        self.output_folder = output_folder
        self.fixed_image_path = fixed_image_path

    def process_files(self):
        """Processes each file in the input path using affine registration."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        files = (os.listdir(self.input_path) if os.path.isdir(self.input_path) 
                 else [self.input_path])
        for file in tqdm(files, desc="Processing files"):
            try:
                moving_image_path = (os.path.join(self.input_path, file)
                                     if os.path.isdir(self.input_path) else file)
                output_path = os.path.join(self.output_folder, file)
                self.affine_registration(moving_image_path, output_path)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    def affine_registration(self, moving_image_path, output_path):
        """Performs affine registration on the given image."""
        fixed_image = ants.image_read(self.fixed_image_path)
        moving_image = ants.image_read(moving_image_path)
        registration = ants.registration(fixed=fixed_image, 
                                         moving=moving_image, 
                                         type_of_transform='Rigid')
        ants.image_write(registration['warpedmovout'], output_path)


def download_model_if_needed(templates_folder):
    """Downloads model from Hugging Face if template folder is empty or doesn't exist."""
    if not os.path.exists(templates_folder) or not os.listdir(templates_folder):
        print("Downloading model from Hugging Face...")
        os.makedirs(templates_folder, exist_ok=True)
        subprocess.run(["huggingface-cli", "download", "hwonheo/easysr_templates", 
                        "--local-dir", "templates", "--local-dir-use-symlinks", "False"], check=True)


def main():
    parser = argparse.ArgumentParser(description="EasySR PreProcessing using AntsPy")
    parser.add_argument("--input", required=True, help="Input NIfTI file or folder")
    parser.add_argument("--t2", action='store_true', help="Use T2 template for fixed image")
    parser.add_argument("--t1", action='store_true', help="Use T1 template for fixed image")
    parser.add_argument("--output", default="train", help="Output folder")
    args = parser.parse_args()

    templates_folder = "templates"
    download_model_if_needed(templates_folder)

    fixed_image_file = "bmc_t1_rat.nii.gz" if args.t1 else "bmc_t2_rat.nii.gz"
    fixed_image_path = os.path.join(templates_folder, fixed_image_file)

    processor = MRIProcessor(args.input, args.output, fixed_image_path)
    processor.process_files()
    print("MRI preprocessing is complete.")

if __name__ == "__main__":
    main()
