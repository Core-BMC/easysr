import os
import sys
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import shutil
import ants
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch
from network.generator import ResnetGenerator
from scipy.ndimage import zoom

# Class to handle MRI image inference
class MRIInference:
    def __init__(self, model, device, input_shape, output_shape):
        # Initialize with model, device, and shapes
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.output_shape = output_shape

    def load_image(self, file_path):
        # Load the image using nibabel
        nib_image = nib.load(file_path)

        image_data = nib_image.get_fdata()
        rotated_image = np.rot90(image_data, k=1, axes=(1, 2))

        # Standard normalization to 0-255
        min_val, max_val = np.min(rotated_image), np.max(rotated_image)
        scale = 255 / (max_val - min_val)
        normalized_image = scale * (rotated_image - min_val)

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
        nib.save(nib.Nifti1Image(resampled_image, np.eye(4)), file_name)

    def match_sform_affine(self, orig_path, gen_path):
        # Match affine transformation of original and generated images
        orig_img = nib.load(orig_path)
        orig_affine = orig_img.affine
        gen_img = nib.load(gen_path)
        gen_data = gen_img.get_fdata()
        matched_gen_img = nib.Nifti1Image(gen_data, orig_affine)
        nib.save(matched_gen_img, gen_path)

    def infer(self, aligned_image_path, original_file_path, output_path):
        # Load and preprocess the image from aligned_image_path
        input_tensor = self.load_image(aligned_image_path)
        
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

        # Reorient the generated image based on original orientation
        gen_img = nib.load(temp_generated_path)
        gen_data = gen_img.get_fdata()
        reoriented_image = ants.from_numpy(gen_data)
        #print(f"Orientation of the original image: {orig_orientation}")  ## print Orientation ##

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

def download_model_if_needed(templates_folder):
    """Downloads model from Hugging Face if template folder is empty or doesn't exist."""
    if not os.path.exists(templates_folder) or not os.listdir(templates_folder):
        print("Downloading model from Hugging Face...")
        os.makedirs(templates_folder, exist_ok=True)
        subprocess.run(["huggingface-cli", "download", "hwonheo/easysr_templates", 
                        "--local-dir", "templates", "--local-dir-use-symlinks", "False"], check=True)

@st.cache_data
def load_model(model_choice):
    # Load pre-trained model based on user selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = ResnetGenerator().to(device)

    if model_choice == "T1-Model":
        checkpoint_path = 'ckpt/ckpt_final/G_latest_T1.pth'
    else: # "Mixed-Model"
        checkpoint_path = 'ckpt/ckpt_final/G_latest_Mixed.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    return generator, device

def run_bias_field_correction(file_path, output_path, correction_type):
    """Bias field correction script and return corrected file path"""
    corrected_file_name = os.path.basename(file_path).replace('.nii', '_corrected.nii')
    corrected_file_path = os.path.join(output_path, corrected_file_name)

    subprocess.run([
        sys.executable, "utils/BiasFieldCorrection.py",
        "--input", file_path,
        "--output", output_path,
        "--type", correction_type
    ])

    # Rename the processed file if necessary
    original_corrected_file_path = os.path.join(output_path, os.path.basename(file_path))
    if os.path.exists(original_corrected_file_path) and original_corrected_file_path != corrected_file_path:
        shutil.move(original_corrected_file_path, corrected_file_path)

    return corrected_file_path

# Perform inference and handle images
def run_inference(inference_engine, aligned_image_path, original_file_path, output_path):
    try:
        # Perform inference using the aligned image and original file path
        warped_image_path = inference_engine.infer(aligned_image_path, original_file_path, output_path)

        # Generate file name for output
        gen_file_name = os.path.basename(original_file_path).replace(".nii", "_gen.nii")
        download_file_path = os.path.join(output_path, gen_file_name)

        # Copy the processed file to the download path
        shutil.copy(warped_image_path, download_file_path)

        # Load original and inferred images for display
        original_img = nib.load(original_file_path).get_fdata()
        inferred_img = nib.load(warped_image_path).get_fdata()

        # Save middle slices of both images for comparison
        original_slice_path = os.path.join(output_path, "original_slice.jpg")
        inferred_slice_path = os.path.join(output_path, "inferred_slice.jpg")
        save_middle_slice(original_img, original_slice_path)
        save_middle_slice(inferred_img, inferred_slice_path)

        # Return paths for UI display
        return (original_slice_path, inferred_slice_path, download_file_path, gen_file_name)
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None, None, None, None

def save_middle_slice(image, file_path):
    # Save the middle slice of the MRI image
    middle_slice = image[image.shape[0] // 2]
    
    # Rotate the image 90 degrees counterclockwise
    rotated_slice = np.rot90(middle_slice)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rotated_slice, cmap='gray', aspect='auto')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(file_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close()

def display_results(original_slice_path, inferred_slice_path, download_file_path, gen_file_name):
    st.subheader("Comparison of Original and EasySR Inferred Slice")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.image(original_slice_path, caption="Original MRI", width=300)
    with col2:
        st.image(inferred_slice_path, caption="Inferred MRI", width=300)

    if os.path.exists(download_file_path):
        with open(download_file_path, "rb") as file:
            st.download_button(
                label="Download (EasySR Inferred-MRI)",
                data=file,
                file_name=gen_file_name,
                mime="application/gzip", 
                type="primary"
            )

def clear_output_folder(folder_path):
    # Clear contents of a specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def clear_session():
    # Clear Streamlit session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Main function for Streamlit UI
def main():
    global original_slice_path, inferred_slice_path, download_file_path, gen_file_name, intensity_adjust

    st.sidebar.markdown("#  ")
    st.sidebar.markdown(
        "[![git](https://img.icons8.com/material-outlined/48/000000/github.png)]"
        "(https://github.com/hwonheo/easysr)"
    )
    st.sidebar.markdown("#  ")

    # Setup sidebar with instructions and model selection
    st.sidebar.subheader("*Model Selection*", divider='red')
    model_choice = st.sidebar.selectbox(
        "Choose the model type:",
        ("Mixed-Model", "T1-Model"),
        index=1  # Default is Combined-Model
    )

    st.sidebar.header("\n")

    # Setup sidebar with instructions
    st.sidebar.subheader("_How to Use EasySR_", divider='red')
    with st.sidebar.expander("Step-by-Step Guide:"):
        st.markdown(
            "1. **Prepare Your Data**: Make sure your rat brain MRI data "
            "is in NIFTI format. Convert if needed.\n\n"
            "2. **Upload Your MRI**: Drag and drop your NIFTI file "
            "or use the upload button.\n\n"
            "3. **Start the EasySR**: Click 'EasySR' to begin processing. "
            "It usually takes a few minutes.\n\n"
            "4. **Sit Back and Relax**: Wait while your data is processed quickly.\n\n"
            "5. **View and Download**: After processing, view the results and "
            "use the download button to save the enhanced MRI data.\n\n"
            "6. **Use as Needed**: Download and utilize your enhanced MRI. "
            "Continue using EasySR for more enhancements.\n\n"
        )
    
    # Initialize model and inference engine with the selected model
    generator, device = load_model(model_choice)
    inference_engine = MRIInference(generator, device, (128, 128, 64), (128, 128, 192))
    
    # Main interface layout
    st.markdown("<h1 style='text-align: center;'>EasySR</h1>", unsafe_allow_html=True)
    st.subheader("_Easy Web UI for Generative 3D Inference of Rat Brain MRI_", divider='red')

    # Initialize paths for processing results
    original_slice_path = None
    inferred_slice_path = None
    download_file_path = None 

    output_path = "infer/generate"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # File uploader for MRI files
    uploaded_file = st.file_uploader("_MRI File Upload (NIFTI)_", 
                                     type=["nii", "nii.gz"], key='file_uploader')
    
    # Checkbox for intensity adjustment
    intensity_adjust = st.checkbox("Bias Field Correction (enhance signal intensity)", 
                                    help="Apply intensity truncation and bias correction to an image: "
                                        "Check this option if the input image exhibits low signal intensity "
                                        "(common in T2RARE, TOF, etc.) or if the output from the inference "
                                        "process appears weakly signaled. This will enhance the signals by "
                                        "N4-bias correction and very low- or high-signal intensity truncation, "
                                        "yielding clearer and more defined results.")

    if uploaded_file is not None:
        # Store uploaded file in session state
        st.session_state['uploaded_file'] = uploaded_file
        file_name = uploaded_file.name

        # Temporary directory for file processing
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, file_name)

        # Write uploaded file to temp directory
        with open(temp_file_path, "wb") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())

        # Inference start button
        if st.button("EasySR (start inference)", type="primary"):
            try:
                # Bias Field Correction
                corrected_file_path = run_bias_field_correction(
                    temp_file_path, temp_dir, "abp") if intensity_adjust else temp_file_path

                # Ensure template files are available
                templates_folder = "templates"
                download_model_if_needed(templates_folder)
                template_path = os.path.join(templates_folder, "bmc_t2_rat.nii.gz")

                # Resample and align the image
                resampled_path = resample_to_isotropic(
                    corrected_file_path, os.path.join(temp_dir, "resampled.nii.gz"))
                aligned_path = align_to_template(
                    resampled_path, template_path, os.path.join(temp_dir, "aligned.nii.gz"))

                # Perform inference and process results
                original_slice_path, inferred_slice_path, download_file_path, gen_file_name = run_inference(
                    inference_engine, aligned_path, corrected_file_path, output_path)

                # Display results
                display_results(original_slice_path, inferred_slice_path, download_file_path, gen_file_name)

            except Exception as e:
                st.error(f"Error during inference: {e}")

    # Button to clear generated content
    if st.button('Clear Generated All', 
            help='Pressing this will delete the contents of the generate folder.'):
        clear_output_folder('infer/generate')
        clear_session()
        st.rerun()

# Entry point for the Streamlit application
if __name__ == '__main__':
    main()

