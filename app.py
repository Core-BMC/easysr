import os
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
        # Load and preprocess MRI image
        image = nib.load(file_path).get_fdata()
        rotated_image = np.rot90(image, k=1, axes=(1, 2))
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
        resampled_image = zoom(normalized_image, scale_factors, order=1)
        return torch.tensor(
            resampled_image[np.newaxis, np.newaxis, ...], dtype=torch.float32)

    def save_image(self, image, file_name):
        # Save processed image to file
        image = image.squeeze().cpu().numpy()
        scale_factors = (
            self.output_shape[0] / image.shape[0], 
            self.output_shape[1] / image.shape[1], 
            self.output_shape[2] / image.shape[2]
        )
        resampled_image = zoom(image, scale_factors, order=1)
        resampled_image = np.rot90(resampled_image, k=-1, axes=(1, 2))
        nib.save(nib.Nifti1Image(resampled_image, np.eye(4)), file_name)

    def match_sform_affine(self, orig_path, gen_path):
        # Match affine transformation of original and generated images
        orig_img = nib.load(orig_path)
        orig_affine = orig_img.affine
        gen_img = nib.load(gen_path)
        gen_data = gen_img.get_fdata()
        matched_gen_img = nib.Nifti1Image(gen_data, orig_affine)
        nib.save(matched_gen_img, gen_path)

    def infer(self, input_tensor, original_file_path, output_path):
        # Perform inference on input tensor
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
        gen_file_name = f"{Path(base_name).stem}_{int(time.time())}_gen.nii.gz"
        warped_file_path = os.path.join(output_path, gen_file_name)
        affine_registration(
            resampled_file_path, resampled_generated_path, warped_file_path)
        for temp_file in [temp_orig_path, temp_generated_path, resampled_generated_path]:
            os.remove(temp_file)
        return warped_file_path
    
    # Perform inference and handle images
    def run_inference(input_tensor, temp_file_path, output_path):
        try:
            warped_image_path = inference_engine.infer(
                input_tensor, temp_file_path, output_path)

            gen_file_name = temp_file_path.replace(".nii", "_gen.nii")
            download_file_path = os.path.join(output_path, gen_file_name)
            shutil.copy(warped_image_path, download_file_path)

            original_img = nib.load(temp_file_path).get_fdata()
            inferred_img = nib.load(warped_image_path).get_fdata()

            original_slice_path = os.path.join(output_path, "original_slice.jpg")
            inferred_slice_path = os.path.join(output_path, "inferred_slice.jpg")
            save_middle_slice(original_img, original_slice_path)
            save_middle_slice(inferred_img, inferred_slice_path)

            return (original_slice_path, inferred_slice_path, 
                    download_file_path, gen_file_name)
        except Exception as e:
            st.error(f"Error during inference: {e}")
            return None, None, None, None

# Image processing functions
def resample_to_isotropic(image_path, output_path):
    # Resample image to isotropic resolution
    image = ants.image_read(image_path)
    resampled_image = ants.resample_image(
        image, (0.15, 0.15, 0.15), use_voxels=False, interp_type=4)
    ants.image_write(resampled_image, output_path)
    return output_path

def affine_registration(fixed_image_path, moving_image_path, output_path):
    # Perform affine registration between two images
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    registration = ants.registration(
        fixed=fixed_image, moving=moving_image, 
        type_of_transform='Elastic')
    ants.image_write(registration['warpedmovout'], output_path)

@st.cache_data
def load_model():
    # Load pre-trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = ResnetGenerator().to(device)
    checkpoint_path = 'ckpt/ckpt_final/G_latest.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    return generator, device

# Initialize model and inference engine
generator, device = load_model()
inference_engine = MRIInference(generator, device, (128, 32, 128), (128, 192, 128))

def save_middle_slice(image, file_path):
    # Save the middle slice of the MRI image
    middle_slice = image[image.shape[0] // 2]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(middle_slice, cmap='gray', aspect='auto')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(file_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close()

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
    global original_slice_path, inferred_slice_path, download_file_path, gen_file_name

    # Setup sidebar with instructions
    st.sidebar.subheader("_How to Use EasySR_", divider='red')
    st.sidebar.markdown(
        "**Step-by-Step Guide:**\n\n"
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
        ":rocket: :red[*EasySR*] \t [Github](https://github.com/hwonheo/easysr)\n\n"
        ":hugging_face: :orange[*EasySR*] \t [Huggingface](https://huggingface.co/spaces/hwonheo/easysr)"
    )

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

    if uploaded_file is not None:
        # Store uploaded file in session state
        st.session_state['uploaded_file'] = uploaded_file
        file_name = uploaded_file.name

        # Inference start button
        infer_button = st.button("EasySR (start inference)", type="primary")

        if infer_button:
            # Temporary directory for file processing
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, file_name)

            # Write uploaded file to temp directory
            with open(temp_file_path, "wb") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())

            # Load image and start inference in a thread
            input_tensor = inference_engine.load_image(temp_file_path)

            def inference_wrapper():
                # Running inference and processing results
                global original_slice_path, inferred_slice_path, download_file_path, gen_file_name
                try:
                    warped_image_path = inference_engine.infer(
                        input_tensor, temp_file_path, output_path)
                    gen_file_name = file_name.replace(".nii", "_gen.nii")
                    download_file_path = os.path.join(output_path, gen_file_name)
                    shutil.copy(warped_image_path, download_file_path)

                    # Load original and inferred images for display
                    original_img = nib.load(temp_file_path).get_fdata()
                    inferred_img = nib.load(warped_image_path).get_fdata()
                    original_slice_path = os.path.join(output_path, "original_slice.jpg")
                    inferred_slice_path = os.path.join(output_path, "inferred_slice.jpg")

                    # Save middle slice of both images for comparison
                    save_middle_slice(original_img, original_slice_path)
                    save_middle_slice(inferred_img, inferred_slice_path)
                except Exception as e:
                    st.error(f"Error during inference: {e}")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

            # Start thread for inference
            inference_thread = threading.Thread(target=inference_wrapper)
            inference_thread.start()

            # Display spinner while processing
            with st.spinner("Processing your MRI image..."):
                inference_thread.join()

        # Display comparison images and download button after processing
        if original_slice_path and os.path.exists(original_slice_path) \
                and inferred_slice_path and os.path.exists(inferred_slice_path):
            st.subheader("Comparison of Original and EasySR Inferred Slice")
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.markdown("**Original**")
                st.image(original_slice_path, caption="Original MRI", width=300)
            with col2:
                st.markdown("**EasySR**")
                st.image(inferred_slice_path, caption="Inferred MRI", width=300)

        if download_file_path and os.path.exists(download_file_path):
            with open(download_file_path, "rb") as file:
                st.download_button(
                    label="Download (EasySR Inferred-MRI)",
                    data=file,
                    file_name=gen_file_name,
                    mime="application/gzip",
                    type="primary"
                )

        # Button to clear generated content
        if st.button('Clear Generated All', 
            help='Pressing this will delete the contents of the generate folder.'):
            clear_output_folder('infer/generate')
            clear_session()
            st.rerun()

# Entry point for the Streamlit application
if __name__ == '__main__':
    main()

