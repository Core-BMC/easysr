#EasySR

![image](https://github.com/hwonheo/easysr/assets/109127356/afdc6f41-8b48-4ca8-b294-cf7f81a9157e)


- Overview

    **EasySR** is a cutting-edge project focused on **super-resolution reconstruction of rat brain MR images**. This project's training code specifically addresses the challenge of enhancing axial slice resolution, commonly recorded as thicker slices in rat brain MRIs. Our approach is tailored to work with high-precision isovoxel rat brain MRI data, captured at a fine resolution of 0.2mm or less.

    The primary goal of EasySR is to upscale these images to an even finer resolution of 0.15mm, achieving an isotropic output. This enhancement allows for more detailed and precise anatomical studies. However, it's important to note that while EasySR excels in spatial resolution improvement, it does not maintain the original signal intensity due to the normalization process involved. Therefore, this tool may not be suitable for experiments that rely heavily on signal intensity measurements.

    Currently, EasySR is in an experimental stage and requires extensive validation. The effectiveness of the super-resolution reconstruction heavily depends on the quality and quantity of the training data. We recommend using at least five high-quality datasets for training to achieve optimal results. Users are encouraged to train the model using the provided pre-trained checkpoints and to contribute back by sharing their well-trained checkpoints. This collaborative approach will significantly enhance the utility and accuracy of EasySR for the scientific community.

    As this project does not involve data collection, we rely on contributions from users for improved training datasets. **With more data and shared checkpoints, EasySR has the potential to become a more robust and reliable tool for rat brain MRI analysis.**

#

- Key Features
1. PyTorch-Based Architecture: The core of EasySR is built using PyTorch, a leading deep learning framework that ensures high performance and flexibility.

2. Custom Dataset Class for MRI: The MRIDataset class handles the loading, processing, and augmentation of MRI images. This includes resampling for different resolutions, normalization, and optional transformations like random affine shifts and flips.

3. Advanced GAN Structure: Incorporates a Generative Adversarial Network (GAN) with a specialized Resnet-based generator (ResnetGenerator) and a patch-based discriminator (PatchDiscriminator). This structure is optimized for MRI super-resolution tasks.

4. Dynamic Augmentation: The training process supports dynamic augmentation, allowing the model to learn from a diverse range of transformations, thereby improving its generalization capabilities.

5. Efficient Image Processing: Utilizes TorchIO and Nibabel for efficient handling and processing of MRI data, ensuring compatibility with standard medical imaging formats.

6. Automatic CUDA Optimization: Includes a function to automatically check for CUDA availability and leverages GPU acceleration for training, making the process faster and more efficient.

7. Customizable Training Loop: The training loop is designed to be flexible, supporting adjustable parameters like epochs, batch size, and learning rate. It also includes detailed logging for monitoring the training process.

8. Checkpointing and Loss Tracking: Implements a sophisticated system for saving model checkpoints and tracking loss, which is crucial for long training sessions and iterative model improvement.

9. Image Visualization Tools: The script includes functions to save and visualize the middle slice of the MRI images, providing an easy way to inspect the results of the super-resolution process.

10. Gradient Scaling and Mixed Precision Training: Utilizes GradScaler and autocast from torch.cuda.amp for mixed precision training, which can lead to faster computations and reduced memory usage while maintaining the model's accuracy.

#


***Requirements***

    Python 3.x
    PyTorch
    

  
#

***Installation***



```bash
git clone https://github.com/hwonheo/easysr.git
cd easysr
pip install -r requirements.txt
```

#

***Usage (Train.py)***


Training own your data 

```bash
python train.py --epochs 100 --batch_size 4 --save_path './ckpt'
```
or, training on going your new data in train folder
```bash
python train.py --epochs 100 --batch_size 4 --final './ckpt'
```
or, training on going your data in train folder using provided pre-train checkpoint
```bash
python train.py --epochs 100 --batch_size 4 --final
```
'--help' provide more info
```bash
python train.py --help
```

#

**Dataset Preparation**

To effectively train the EasySR model, it's essential to prepare your dataset with specific requirements in mind. Ensure your data meets the following criteria for optimal results:

1. Resolution and Type: The dataset should consist of high-resolution rat brain MRI images. Each image must be captured with a resolution of **0.2mm** or finer to ensure the model can effectively learn and enhance the image quality. This fine resolution is crucial for achieving the desired super-resolution outcomes.

2. Isotropy: The images should be **isovoxel**, meaning they maintain equal resolution in all three dimensions. This isotropy is key for the model to uniformly upscale the images in the axial plane without distorting the anatomy.

3. File Format: Prepare your data in the **NIfTI format**, which is widely used in medical imaging. The files should be either in .nii or .nii.gz format. This standard format ensures compatibility with the data loading and processing methods used in the EasySR code.

4. Dataset Size: For effective training, it's recommended to use a minimum of five distinct datasets. This diversity in the training data helps the model generalize better and enhances its ability to upscale various images accurately.

5. Data Organization: Organize your dataset in a structured manner, preferably in a dedicated directory. This organization facilitates easier loading and batch processing of the images during training.

Ensure that your dataset is prepared with care, as the quality of the training data significantly influences the performance of the EasySR model. By adhering to these guidelines, you set the foundation for successful super-resolution reconstruction of rat brain MRI images.

#

**Code Structure**

1. ResnetBlock
- Purpose: A fundamental building block of the ResnetGenerator, designed to create a convolutional block with residual learning.
Implementation 
- Details:
  - Each ResnetBlock consists of two convolutional layers (nn.Conv3d), each followed by batch normalization (nn.BatchNorm3d) and the first followed by a LeakyReLU activation (nn.LeakyReLU).
  - The block implements a skip connection by adding the input to the output of the convolutional block, facilitating gradient flow and mitigating the vanishing gradient problem in deep networks.
  - This structure is crucial for learning identity mappings and refining features across the network.
2. DeUpBlock
- Purpose: Specifically designed for upsampling in the width dimension, this block is integral for increasing the resolution of the MRI slices in the desired dimension.
- Implementation Details:
  - The DeUpBlock uses a 3D transposed convolution (nn.ConvTranspose3d) for upsampling. This approach is tailored to upsample the MRI images selectively along the width dimension, aligning with the project's focus on axial slice super-resolution.
  - A LeakyReLU activation follows the transposed convolution, adding non-linearity to the upscaling process.
  - This selective upsampling is a unique aspect of the EasySR project, differentiating it from typical 3D upsampling methods that uniformly scale across all dimensions.
3. PatchDiscriminator
- The PatchDiscriminator class in the EasySR project is a key component of the adversarial network, designed to differentiate between real and generated super-resolution MRI images.
- Initialization: The discriminator takes grayscale images as input (input_nc=1) and begins with a relatively small number of filters (ndf=16). This setting is optimal for processing the MRI data.
- Convolutional Layers:
  - The model consists of several convolutional layers, each forming a conv_block. These blocks are composed of a 3D convolution (nn.Conv3d), batch normalization (nn.BatchNorm3d), and LeakyReLU activation (nn.LeakyReLU), progressively increasing the depth of the feature maps.
  - The series of convolutions (from conv1 to conv4) gradually downsample the input, extracting increasingly abstract and complex features from the images.
  - The final convolution layer (conv5) is designed to reduce the output to a single channel, setting the stage for binary classification.
 - Output Layer:
   - Following the convolutional layers, the model employs a flatten layer (nn.Flatten) and a fully connected layer (nn.Linear). The number '539' in the linear layer's input should be adjusted based on the flattened output size of the preceding layers.
   - The final output is obtained through a sigmoid activation function (nn.Sigmoid), providing a probability score indicating whether the input image is real or generated.
 - Purpose and Function:
   - The PatchDiscriminator is tailored to assess localized regions (or 'patches') of the input images, making it particularly effective for tasks like super-resolution where fine details are crucial.
   - By distinguishing between real and upscaled images, this discriminator plays a crucial role in training the generator to produce more realistic super-resolution outputs.


#

**Inference Process**

The EasySR project includes a comprehensive inference script that allows users to apply the trained ResnetGenerator model to their own MRI images for super-resolution reconstruction. Here's an overview of how this process works:

**Key Components:**

*MRIInference Class:* A dedicated class for handling the inference process. It loads the MRI images, processes them for input into the model, and saves the output images.

*Image Resampling:* The script includes functionality to resample input images to isotropic resolution using the ants library, ensuring consistency in image dimensions before feeding them to the model.

*Affine Registration:* Post-processing includes an affine registration step to align the generated image with the original, maintaining anatomical accuracy.


**Process Workflow:**

*Loading the Model:* The script initializes the ResnetGenerator model and loads the pre-trained weights from a specified checkpoint.

*Image Processing:* For each input MRI image, the script performs normalization, rotation, and resampling to match the model's input shape requirements.
Model Inference: The processed image is fed into the model, generating a super-resolved output.

*Post-Processing:*
The output image is resampled back to the desired isotropic resolution.
Affine registration aligns the generated image with the original MRI scan, facilitating direct comparison.

*Saving the Result:* The final super-resolution image is saved, providing a detailed and enhanced view of the original MRI scan.

**Usage:**

To perform inference, users can run the script with the following command, specifying the paths to their input MRI images, the model checkpoint, and the desired output directory:

```bash
python inference.py --input [input_path] --ckpt [checkpoint_path] --output [output_path]
```

#

**WebUI Inference (Streamlit)**

Please visit, [EasySR HF-space](https://huggingface.co/spaces/hwonheo/easysr)

#

**Reference**

https://github.com/imatge-upc/3D-GAN-superresolution  


#



**Contributing**

https://github.com/hwonheo  
https://github.com/ssimu  
https://github.com/olivepicker  


#

**License**

not yet.

#

**Contact**

heohwon@gmail.com

#

**Acknowledgements**

This research was supported by Basic Science Research Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Education(NRF-2022R1I1A1A01072397)
