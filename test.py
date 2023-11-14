import os
import shutil

def copy_specific_files(src_directory):
    """
    Copies files that match the specific format (subfolder name equals file name with .txt extension)
    from subfolders of the given source directory to a target directory named 'scannet_scene_info'
    located in the parent directory of the source directory.

    :param src_directory: Path to the source directory (Folder A).
    """
    # Define the target directory
    parent_directory = os.path.dirname(src_directory)
    target_directory = os.path.join(parent_directory, 'scannet_scene_info')

    # Create the target directory if it does not exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Walk through the subdirectories of the source directory
    for subdir, _, files in os.walk(src_directory):
        subdir_name = os.path.basename(subdir)
        for file in files:
            # Check if the file name matches the format (subfolder_name.txt)
            if file == f'{subdir_name}.txt':
                src_file_path = os.path.join(subdir, file)
                target_file_path = os.path.join(target_directory, file)

                # Copy the file to the target directory
                shutil.copy(src_file_path, target_file_path)

# Usage example:
# Assuming your current working directory is the Folder A
# current_directory = os.getcwd()
current_directory = "H:\ScanNet_Data\data\scannet\scans"
copy_specific_files(current_directory)

# You need to replace 'current_directory' with the actual path of Folder A when you run the script.

