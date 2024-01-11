import os
import shutil
import random
import argparse

class DataSplitter:
    """Class to split Nii or nii.gz files into training and validation datasets."""

    def __init__(self, input_folder, train_ratio, train_folder, val_folder):
        """
        Initialize the DataSplitter with folder paths and data ratio.

        :param input_folder: Path to the folder containing the Nii files.
        :param train_ratio: Ratio of the dataset to be used as training data.
        :param train_folder: Name of the folder to store training data.
        :param val_folder: Name of the folder to store validation data.
        """
        self.input_folder = input_folder
        self.train_ratio = train_ratio
        self.train_folder = os.path.join(input_folder, train_folder)
        self.val_folder = os.path.join(input_folder, val_folder)

    def split_data(self):
        """
        Split the data into training and validation datasets and move files
        to the respective folders.
        """
        # List all Nii or nii.gz files in the input folder
        files = [f for f in os.listdir(self.input_folder)
                 if f.endswith('.nii') or f.endswith('.nii.gz')]
        random.shuffle(files)  # Shuffle files for random splitting

        # Determine the split index based on the training ratio
        split_index = int(len(files) * self.train_ratio)
        train_files = files[:split_index]
        val_files = files[split_index:]

        # Create training and validation folders if they don't exist
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.val_folder, exist_ok=True)

        # Move files to the respective folders
        for file in train_files:
            shutil.move(os.path.join(self.input_folder, file),
                        self.train_folder)

        for file in val_files:
            shutil.move(os.path.join(self.input_folder, file),
                        self.val_folder)

        print(f"Files split into {len(train_files)} training "
              f"and {len(val_files)} validation.")

def main():
    """
    Main function to handle command line arguments and initiate data splitting.
    """
    parser = argparse.ArgumentParser(
        description='Split Nii files into training and validation datasets.')

    # Define command line arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input folder with Nii files.')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training data ratio (default: 0.8).')
    parser.add_argument('--train-folder', type=str, default='training',
                        help='Folder for training data (default: "training").')
    parser.add_argument('--val-folder', type=str, default='validation',
                        help='Folder for validation data (default: "validation").')

    args = parser.parse_args()

    # Create a DataSplitter instance and split the data
    splitter = DataSplitter(args.input, args.train_ratio,
                            args.train_folder, args.val_folder)
    splitter.split_data()

if __name__ == "__main__":
    main()
