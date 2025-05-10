import os
import shutil

def delete_files(folder):
    """
    Deletes all files in the specified folder.

    :param folder: Path to the folder from which you want to delete files.
    """
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return

    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"Skipped: {file_path} (not a file)")

    print("File deletion process completed.")

def delete_folder(folder):
    """
    Deletes the specified folder and all its contents.

    :param folder: Path to the folder to be deleted.
    """
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return

    if not os.path.isdir(folder):
        raise ValueError(f"Path {folder} is not a directory.")

    shutil.rmtree(folder)
    print(f"Deleted folder: {folder}")

def clear(keras=False, output=False, training_masks=False, training_pictures=False, input_unet=False, all=False, dry_run=False):
    """
    Clears specified directories and files based on provided flags. If 'all' is True, clears all specified locations.

    :param keras: If True, deletes files with '.keras' extension in the current directory.
    :param output: If True, deletes the 'output_contours' folder.
    :param training_masks: If True, deletes all files in the 'training/mask' folder.
    :param training_pictures: If True, deletes all files in the 'training/original' folder.
    :param input_unet: If True, deletes all files in the 'input_unet' folder.
    :param all: If True, performs all actions regardless of individual flags.
    :param dry_run: If True, lists actions to be performed without making changes.
    """
    if all:
        keras = output = training_masks = training_pictures = input_unet = True

    actions_summary = []

    if keras:
        keras_files = [file_name for file_name in os.listdir('.') if file_name.endswith('.keras')]
        if dry_run:
            actions_summary.extend([f"Would delete: {file}" for file in keras_files])
        else:
            for file_name in keras_files:
                os.remove(file_name)
                print(f"Deleted file: {file_name}")

    if output:
        if dry_run:
            actions_summary.append(f"Would delete folder: output_contours")
        else:
            delete_folder('output_contours')

    if training_masks:
        if dry_run:
            files = os.listdir('training/mask') if os.path.exists('training/mask') else []
            actions_summary.extend([f"Would delete: training/mask/{file}" for file in files])
        else:
            delete_files('training/mask')

    if training_pictures:
        if dry_run:
            files = os.listdir('training/original') if os.path.exists('training/original') else []
            actions_summary.extend([f"Would delete: training/original/{file}" for file in files])
        else:
            delete_files('training/original')

    if input_unet:
        if dry_run:
            files = os.listdir('input_unet') if os.path.exists('input_unet') else []
            actions_summary.extend([f"Would delete: input_unet/{file}" for file in files])
        else:
            delete_files('input_unet')

    if dry_run:
        print("Dry run summary:")
        for action in actions_summary:
            print(action)

if __name__ == "__main__":
    clear(keras=True, output=True, dry_run=True)
