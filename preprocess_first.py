import os
import shutil

# Set your source and destination directories
source_dir = '../batch_20231027_part1/Part1'
destination_dir = '../batch_20231027_part1/Part1/Preprocessed'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


# Function to check and copy valid rows
def copy_valid_rows(row_number):
    valid = True
    files_to_copy = []

    # Ground truth image and parameters file
    gt_image = f'0_{row_number}_GT_pose_0_thermal.png'
    param_file = f'0_{row_number}_Parameters.txt'

    # Check if ground truth image and parameters file exist
    if not (os.path.exists(os.path.join(source_dir, gt_image)) and os.path.exists(os.path.join(source_dir, param_file))):
        valid = False

    # Check if all 11 images exist
    for i in range(11):
        img_name = f'0_{row_number}_pose_{i}_thermal.png'
        if os.path.exists(os.path.join(source_dir, img_name)):
            files_to_copy.append(img_name)
        else:
            valid = False
            break

    # Create folder and copy files if row is valid
    if valid:
        row_folder = os.path.join(destination_dir, f'row_{row_number}')
        os.makedirs(row_folder)

        # Subfolder for the 11 images
        images_subfolder = os.path.join(row_folder, 'Images')
        os.makedirs(images_subfolder)

        # Copy ground truth image and parameters file to the row folder
        shutil.copy(os.path.join(source_dir, gt_image), row_folder)
        shutil.copy(os.path.join(source_dir, param_file), row_folder)

        # Copy each image to the images subfolder
        for file in files_to_copy:
            shutil.copy(os.path.join(source_dir, file), images_subfolder)

# Iterate through each row and process
for row in range(5550):
    copy_valid_rows(row)

print("Copying of valid rows complete.")
