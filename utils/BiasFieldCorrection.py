import argparse
import os
import glob
import ants

def abp_n4(image, intensity_truncation=(0.025, 0.975, 256), mask=None, 
           use_n3=False):
    """
    Apply intensity truncation and bias correction to an image.

    Parameters:
    - image: Image to be processed (ANTsImage object).
    - intensity_truncation: Tuple (lower quantile, upper quantile, number of bins).
    - mask: Optional mask for bias correction (ANTsImage object).
    - use_n3: Use N3 bias correction instead of N4 if True.

    Returns:
    - Processed image (ANTsImage object).
    """
    if not isinstance(intensity_truncation, (list, tuple)) or \
       len(intensity_truncation) != 3:
        raise ValueError("intensity_truncation must be list/tuple with 3 values")

    # Apply intensity truncation
    truncated_image = ants.iMath(image, "TruncateIntensity", 
                                 intensity_truncation[0], intensity_truncation[1],
                                 intensity_truncation[2])

    # Apply bias correction
    if use_n3:
        corrected_image = ants.n3_bias_field_correction(truncated_image, mask=mask)
    else:
        corrected_image = ants.n4_bias_field_correction(truncated_image, mask=mask)

    return corrected_image


def preprocess_image(file_path, output_path, preprocess_type):
    """
    Read, preprocess, and write an image based on the specified method.

    Parameters:
    - file_path: Path to the input image file.
    - output_path: Path to the output folder.
    - preprocess_type: Preprocessing method (n3, n4, or abp).
    """
    image = ants.image_read(file_path)

    if preprocess_type == "n3":
        processed_image = ants.n3_bias_field_correction(image)
    elif preprocess_type == "n4":
        processed_image = ants.n4_bias_field_correction(image)
    elif preprocess_type == "abp":
        processed_image = abp_n4(image)
    else:
        raise ValueError("Invalid preprocess type")

    ants.image_write(processed_image, os.path.join(output_path, 
                                                   os.path.basename(file_path)))


def main():
    """
    Main function to handle command line arguments and process images.
    """
    parser = argparse.ArgumentParser(
        description="Image Preprocessing Script for Bias Field Correction")
    parser.add_argument('--input', type=str, required=True, 
                        help="Input file or folder path")
    parser.add_argument('--output', type=str, default='output', 
                        help="Output folder path")
    parser.add_argument('--type', type=str, default='abp', 
                        choices=['n3', 'n4', 'abp'], 
                        help="Type of preprocessing (n3, n4, abp)")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    files = [args.input] if os.path.isfile(args.input) else \
            glob.glob(os.path.join(args.input, '*'))

    for file in files:
        preprocess_image(file, args.output, args.type)

    print("----- Bias Field Correction Process completed -----")


if __name__ == '__main__':
    main()
