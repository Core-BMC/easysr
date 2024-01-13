import argparse
import glob
import os
import ants

class MRIRegistrationPreprocessor:
    """
    A class for preprocessing 3D MRI images using registration-based methods.
    """
    def __init__(self, directory=None, axial_path=None, sagittal_path=None, coronal_path=None, output_directory="output"):
        self.directory = directory
        self.axial_path = axial_path
        self.sagittal_path = sagittal_path
        self.coronal_path = coronal_path
        self.output_directory = output_directory

    def find_images_in_directory(self):
        """
        Find MRI images in the given directory based on their naming patterns.
        """
        if self.directory:
            ax_images = glob.glob(os.path.join(self.directory, '*ax*.nii*'))
            sag_images = glob.glob(os.path.join(self.directory, '*sag*.nii*'))
            cor_images = glob.glob(os.path.join(self.directory, '*cor*.nii*'))

            self.axial_path = ax_images[0] if ax_images else None
            self.sagittal_path = sag_images[0] if sag_images else None
            self.coronal_path = cor_images[0] if cor_images else None

    def find_smallest_voxel_size(self, image):
        """
        Find the smallest voxel size in the given image.
        """
        return min(image.spacing)

    def make_voxel_size_isotropic(self, image):
        """
        Adjust the image to have isotropic voxel sizes based on the smallest
        voxel dimension.
        """
        target_voxel_size = self.find_smallest_voxel_size(image)
        new_spacing = [target_voxel_size] * len(image.spacing)
        return ants.resample_image(image, new_spacing, use_voxels=False)

    def resample_with_trilinear(self, image, reference_image):
        """
        Resample the image with trilinear interpolation to match the reference
        image voxel sizes.
        """
        isotropic_image = self.make_voxel_size_isotropic(image)
        return ants.resample_image_to_target(isotropic_image, reference_image,
                                             interp_type=2)

    def affine_registration(self, fixed_image, moving_image):
        """
        Perform affine registration on the moving image to align with the
        fixed image.
        """
        registration = ants.registration(
            fixed=fixed_image, moving=moving_image, type_of_transform='Affine')
        return registration['warpedmovout']

    def process_images(self):
        """
        Main function to process the MRI images. It handles the alignment
        and averaging of the images.
        """
        if not self.axial_path and not self.coronal_path:
            return "No axial or coronal images provided. No alignment performed."

        reference_path = self.axial_path if self.axial_path else self.coronal_path
        reference_image = ants.image_read(reference_path)
        isotropic_reference = self.make_voxel_size_isotropic(reference_image)
        resampled_reference = self.resample_with_trilinear(isotropic_reference, isotropic_reference)

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        output_file_path = os.path.join(self.output_directory, "synth_registered.nii.gz")
        images_for_averaging = [resampled_reference]

        for moving_path in [self.sagittal_path, self.coronal_path]:
            if moving_path:
                moving_image = ants.image_read(moving_path)
                isotropic_moving = self.make_voxel_size_isotropic(moving_image)
                aligned_image = self.affine_registration(resampled_reference, isotropic_moving)
                images_for_averaging.append(aligned_image)

        if len(images_for_averaging) > 1:
            average_image = ants.average_images(images_for_averaging)
            ants.image_write(average_image, output_file_path)
            return "Average image computed and saved."

        return "Insufficient images for averaging."


def main():
    parser = argparse.ArgumentParser(description='3D MRI Registration-based Preprocessing Tool')
    parser.add_argument('--dir', '-d', help='Directory containing MRI images')
    parser.add_argument('--axial', '-ax', help='Path to the axial image')
    parser.add_argument('--sagittal', '-sag', help='Path to the sagittal image')
    parser.add_argument('--coronal', '-cor', help='Path to the coronal image')
    parser.add_argument('--output', '-o', help='Output directory for the processed images', default='output')

    args = parser.parse_args()

    preprocessor = MRIRegistrationPreprocessor(
        directory=args.dir,
        axial_path=args.axial,
        sagittal_path=args.sagittal,
        coronal_path=args.coronal,
        output_directory=args.output
    )

    preprocessor.find_images_in_directory()
    result = preprocessor.process_images()
    print(result)

if __name__ == "__main__":
    main()