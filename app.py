import streamlit as st
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from network.generator import ResnetGenerator
import ants
import tempfile
import os
import matplotlib.pyplot as plt
import shutil

# Class for handling MRI inference
class MRIInference:
    def __init__(self, model, device, input_shape, output_shape):
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self.output_shape = output_shape

    def load_image(self, file_path):
        # Load and preprocess the MRI image
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
        # Save the processed image
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
        # Match the affine of original and generated images
        orig_img = nib.load(orig_path)
        orig_affine = orig_img.affine
        gen_img = nib.load(gen_path)
        gen_data = gen_img.get_fdata()
        matched_gen_img = nib.Nifti1Image(gen_data, orig_affine)
        nib.save(matched_gen_img, gen_path)

    def infer(self, input_tensor, original_file_path, output_path):
        # Inference process
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

# Functions for image processing and Streamlit UI handling
def resample_to_isotropic(image_path, output_path):
    image = ants.image_read(image_path)
    resampled_image = ants.resample_image(
        image, (0.15, 0.15, 0.15), use_voxels=False, interp_type=4)
    ants.image_write(resampled_image, output_path)
    return output_path


def affine_registration(fixed_image_path, moving_image_path, output_path):
    # Register affine of fixed and moving images
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    registration = ants.registration(
        fixed=fixed_image, moving=moving_image, 
        type_of_transform='Elastic')
    ants.image_write(registration['warpedmovout'], output_path)

@st.cache_data
def load_model():
    # Load the trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = ResnetGenerator().to(device)
    checkpoint_path = 'ckpt/ckpt_final/G_latest.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    return generator, device

generator, device = load_model()
inference_engine = MRIInference(generator, device, (128, 32, 128), (128, 192, 128))

def save_middle_slice(image, file_path):
    # Save middle slice of the MRI image
    middle_slice = image[image.shape[0] // 2]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(middle_slice, cmap='gray', aspect='auto')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(file_path, format='jpg', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close()

def clear_output_folder(folder_path):
    # Clear contents of the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def clear_session():
    # Clear the session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def main():
    # Sidebar - How to Use Guide
    st.sidebar.title("How to Use EasySR")
    st.sidebar.markdown(
        "**Step-by-Step Guide:**\n\n"
        "1. **Prepare Your Data**: \n\n\tGet your rat brain T2 MRI data. "
        "Ensure it's in NIFTI format. Convert if necessary.\n\n"
        "2. **Upload Your MRI**: \n\n\tDrag and drop your NIFTI file or use "
        "the upload button.\n\n"
        "3. **Start the EasySR**: \n\n\tPress 'EasySR' and we'll handle the rest. "
        "The process is quick, typically taking just a few seconds to complete!\n\n"
        "4. **Sit Back and Relax**: \n\n\tNo long waits here - your data will be "
        "processed in under a minute.\n\n"
        "5. **View and Download**: \n\n\tAfter processing, view the results and "
        "use the download button to save the MRI.\n\n"
        "6. **Use as Needed**: \n\n\tDownload and use your enhanced MRI as you see fit. "
        "Get your data more!\n\n  #"
        "#\n\n  "
        "#\n\n\n  "
        "#\n\n\n  "
        "GitHub: EasySR"
        "\n\n  "
        "[github.com/hwonheo/easysr](https://github.com/hwonheo/easysr)"
        "\n\n  "
        "Huggingface (space): EasySR"
        "\n\n  "
        "[huggingface.co/spaces/hwonheo/easysr]"
        "(https://huggingface.co/spaces/hwonheo/easysr)"
    )

    # Main function for Streamlit UI
    st.markdown("<h1 style='text-align: center;'>EasySR:</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Rat Brain T2 MRI SR-Reconstruction</h2>", unsafe_allow_html=True)
    st.title("\n")
    col1, col2 = st.columns([0.5, 0.5])

    original_slice_path = None
    inferred_slice_path = None
    download_file_path = None 

    output_path = "infer/generate"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with col1:
        st.markdown("<h3 style='text-align: center;'>MRI File Upload (NIFTI)</h3>",
                    unsafe_allow_html=True)      
        uploaded_file = st.file_uploader("", type=["nii", "nii.gz"], key='file_uploader')

        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
            file_name = uploaded_file.name
            infer_button = st.button("EasySR (start inference)", type="primary")

            if infer_button:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name

                try:
                    input_tensor = inference_engine.load_image(file_path)
                    warped_image_path = inference_engine.infer(
                        input_tensor, file_path, output_path)

                    gen_file_name = file_name.replace(".nii", "_gen.nii")
                    download_file_path = os.path.join(output_path, gen_file_name)
                    shutil.copy(warped_image_path, download_file_path)

                    original_img = nib.load(file_path).get_fdata()
                    inferred_img = nib.load(warped_image_path).get_fdata()

                    original_slice_path = os.path.join(output_path, "original_slice.jpg")
                    inferred_slice_path = os.path.join(output_path, "inferred_slice.jpg")
                    save_middle_slice(original_img, original_slice_path)
                    save_middle_slice(inferred_img, inferred_slice_path)

                except Exception as e:
                    st.error(f"Error during inference: {e}")

                if file_path and os.path.exists(file_path):
                    os.remove(file_path)

    with col2:
        st.header("\n")
        st.header("\n")
        st.header("\n")
        st.header("\n")
        st.header("\n")
        st.header("\n")
        st.header("\n")
        st.subheader("\n")
        if download_file_path and os.path.exists(download_file_path):
            with open(download_file_path, "rb") as file:
                st.download_button(
                    label="Download (EasySR inferred-MRI)",
                    data=file,
                    file_name=gen_file_name,
                    mime="application/gzip",
                    type="primary"
                )
        
        if st.button('Clear All', 
                     help='Caution: Pressing the Clear All button will delete the contents of the generate folder.'):
            clear_output_folder('infer/generate')
            clear_session()
            st.experimental_rerun()

    st.subheader("\n")
    st.subheader("\n")
    st.subheader("\n")
    if original_slice_path and os.path.exists(original_slice_path):
        st.subheader("Comparison of Inferred slice")
        col3, col4 = st.columns([0.5, 0.5])
        with col3:
            if original_slice_path and os.path.exists(original_slice_path):
                st.markdown("**Original**")
                st.image(original_slice_path, caption="Original MRI", width=300)
        
        with col4:
            if inferred_slice_path and os.path.exists(inferred_slice_path):
                st.markdown("**EasySR**")
                st.image(inferred_slice_path, caption="Inferred MRI", width=300)

if __name__ == '__main__':
    main()