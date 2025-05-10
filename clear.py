import os
import shutil

def delete_files(folder):
    """
    Deletes all files in the specified folder.

    :param folder: Path to the folder from which you want to delete files.
    """
    try:
        # Check if the folder exists
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            return

        # Iterate through all files in the folder
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            # Check if it is a file
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"Skipped: {file_path} (not a file)")

        print("File deletion process completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_folder(folder):
    """
    Deletes the specified folder and all its contents.

    :param folder: Path to the folder to be deleted.
    """
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")
        else:
            print(f"Folder {folder} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def clear(keras=None, output=None, training_masks=None, training_pictures=None, input_unet=None, all=None):
    """
    Clears specified directories and files based on provided flags. If 'all' is True, clears all specified locations.

    :param keras: If True, deletes files with '.keras' extension in the current directory.
    :param output: If True, deletes the 'output_contours' folder.
    :param training_masks: If True, deletes all files in the 'training/mask' folder.
    :param training_pictures: If True, deletes all files in the 'training/original' folder.
    :param input_unet: If True, deletes all files in the 'input_unet' folder.
    :param all: If True, performs all actions regardless of individual flags.
    """
    if all:
        if keras is None:
            keras = True
        if output is None:
            output = True
        if training_masks is None:
            training_masks = True
        if training_pictures is None:
            training_pictures = True
        if input_unet is None:
            input_unet = True

    if keras:
        try:
            for file_name in os.listdir('.'):  # Current directory
                if file_name.endswith('.keras'):
                    os.remove(file_name)
                    print(f"Deleted file: {file_name}")
        except Exception as e:
            print(f"An error occurred while deleting .keras files: {e}")

    if output:
        delete_folder('output_contours')

    if training_masks:
        delete_files('training/mask')

    if training_pictures:
        delete_files('training/original')

    if input_unet:
        delete_files('input_unet')

if __name__ == "__main__":
    clear(keras=True, output=True)
