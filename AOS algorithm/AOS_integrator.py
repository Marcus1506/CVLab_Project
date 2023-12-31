
## Import libraries section ##
import numpy as np
import cv2
import os
import math
from LFR_utils import read_poses_and_images,pose_to_virtualcamera, init_aos, init_window
import LFR_utils as utils
import pyaos
import glm
import glob
import shutil

# p
# ath to where the results will be stored
Download_Location = os.path.dirname(os.path.realpath(__file__))
training_data_folder = os.path.join(Download_Location, 'dataset_for_training')
if not os.path.exists(training_data_folder):
    os.makedirs(training_data_folder)

# Start the AOS Renderer
w, h, fovDegrees = 512, 512, 50  # resolution and field of view. This should not be changed.
render_fov = 50

if 'window' not in locals() or window == None:
    window = pyaos.PyGlfwWindow(w, h, 'AOS')

aos = pyaos.PyAOS(w, h, fovDegrees)


set_folder = os.path.dirname(os.path.realpath(__file__))          # Enter path to your LFR/python directory
aos.loadDEM(os.path.join(set_folder, 'zero_plane.obj'))

####################################################################################################################

#############################Create Poses for Initial Positions###############################################################

# Below are certain functions required to convert the poses to a certain format to be compatabile with the AOS Renderer.


def eul2rotm(theta):
    s_1 = math.sin(theta[0])
    c_1 = math.cos(theta[0]) 
    s_2 = math.sin(theta[1]) 
    c_2 = math.cos(theta[1]) 
    s_3 = math.sin(theta[2]) 
    c_3 = math.cos(theta[2])
    rotm = np.identity(3)
    rotm[0, 0] = c_1*c_2
    rotm[0, 1] = c_1*s_2*s_3 - s_1*c_3
    rotm[0, 2] = c_1*s_2*c_3 + s_1*s_3

    rotm[1, 0] = s_1*c_2
    rotm[1, 1] = s_1*s_2*s_3 + c_1*c_3
    rotm[1, 2] = s_1*s_2*c_3 - c_1*s_3

    rotm[2, 0] = -s_2
    rotm[2, 1] = c_2*s_3
    rotm[2, 2] = c_2*c_3

    return rotm


def createviewmateuler(eulerang, camLocation):
    rotationmat = eul2rotm(eulerang)
    translVec = np.reshape((-camLocation @ rotationmat),(3,1))
    conjoinedmat = (np.append(np.transpose(rotationmat), translVec, axis=1))
    return conjoinedmat


def divide_by_alpha(rimg2):
    a = np.stack((rimg2[:, :, 3], rimg2[:, :, 3], rimg2[:, :, 3]), axis=-1)
    return rimg2[:, :, :3]/a


def pose_to_virtualcamera(vpose ):
    vp = glm.mat4(*np.array(vpose).transpose().flatten())
    #vp = vpose.copy()
    ivp = glm.inverse(glm.transpose(vp))
    #ivp = glm.inverse(vpose)
    Posvec = glm.vec3(ivp[3])
    Upvec = glm.vec3(ivp[1])
    FrontVec = glm.vec3(ivp[2])
    lookAt = glm.lookAt(Posvec, Posvec + FrontVec, Upvec)
    cameraviewarr = np.asarray(lookAt)
    #print(cameraviewarr)
    return cameraviewarr  



########################## Below we generate the poses for rendering #####################################
# This is based on how renderer is implemented. 

Numberofimages = 11  # Or just the number of images
Focal_plane = 0       # Focal plane is set to the ground so it is zero.

# ref_loc is the reference location or the poses of the images. The poses are the same for the dataset and therefore only the images have to be replaced.

ref_loc = [[5,4,3,2,1,0,-1,-2,-3,-4,-5],[0,0,0,0,0,0,0,0,0,0,0]]   # These are the x and y positions of the images. It is of the form [[x_positions],[y_positions]]

altitude_list = [35,35,35,35,35,35,35,35,35,35,35] # [Z values which is the height]

center_index = 5  # this is important, this will be the pose index at which the integration should happen. For example if you have 5 images, lets say you want to integrate all 5 images to the second image position. Then your center_index is 1 as index starts from zero.

site_poses = []
for i in range(Numberofimages):
    EastCentered = (ref_loc[0][i] - 0.0) #Get MeanEast and Set MeanEast
    NorthCentered = (0.0 - ref_loc[1][i]) #Get MeanNorth and Set MeanNorth
    M = createviewmateuler(np.array([0.0, 0.0, 0.0]),np.array( [ref_loc[0][i], ref_loc[1][i], - altitude_list[i]] ))
    print('m',M)
    ViewMatrix = np.vstack((M, np.array([0.0,0.0,0.0,1.0],dtype=np.float32)))
    print(ViewMatrix)
    camerapose = np.asarray(ViewMatrix.transpose(),dtype=np.float32)
    print(camerapose)
    site_poses.append(camerapose)  # site_poses is a list now containing all the poses of all the images in a certain format that is accecpted by the renderer.


#############################Read the generated images from the simulator and store in a list ###############################################################

import re

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# Define the function to merge three images into a multi-channel image
def merge_integral_images(img_paths):
    # Initialize a list to store the images
    images = []

    # Read and convert each image to grayscale if necessary
    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            raise ValueError(f"Image not found or could not be read: {path}")
        images.append(img)

    # Check if all images are of the same size
    if not all(img.shape == images[0].shape for img in images):
        raise ValueError("All images must be of the same size")

    # Merge the images into a single multi-channel image
    merged_image = cv2.merge(images)

    return merged_image


preprocessed_dir = os.path.join(Download_Location, 'Preprocessed')
row_folders = [d for d in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, d))]

for row_folder in row_folders:
    row_path = os.path.join(preprocessed_dir, row_folder)
    images_subfolder = os.path.join(row_path, 'Images')
    imagelist = []

    # Read images
    for img in sorted(glob.glob(images_subfolder + '/*.png'), key=numericalSort):
        n = cv2.imread(img)
        imagelist.append(n)

    row_index = row_folder.split("_")[-1]

    if len(imagelist) == 11:
        # Integral imaging process for each focal plane
        for focal_plane in [0, -1.5, -3]:
            aos.clearViews()
            for i in range(len(imagelist)):
                aos.addView(imagelist[i], site_poses[i], "DEM BlobTrack")
            aos.setDEMTransform([0, 0, focal_plane])

            proj_RGBimg = aos.render(pose_to_virtualcamera(site_poses[center_index]), render_fov)
            tmp_RGB = divide_by_alpha(proj_RGBimg)

            # Save integral image with focal plane value in the name
            integral_image_name = f'2_{row_index}_integral_fp{focal_plane}.png'
            cv2.imwrite(os.path.join(training_data_folder, integral_image_name), tmp_RGB)

        integral_image_paths = [
            os.path.join(training_data_folder, f'2_{row_index}_integral_fp0.png'),
            os.path.join(training_data_folder, f'2_{row_index}_integral_fp-1.5.png'),
            os.path.join(training_data_folder, f'2_{row_index}_integral_fp-3.png')
        ]

        # Merge the integral images
        merged_image = merge_integral_images(integral_image_paths)
        # Save the merged image
        merged_image_path = os.path.join(training_data_folder, f'2_{row_index}_merged_integral.png')
        cv2.imwrite(merged_image_path, merged_image)

        # Delete the original integral images
        for img_path in integral_image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)

        # Copy ground truth and parameters file to the training data folder
        gt_image = f'2_{row_folder.split("_")[-1]}_GT_pose_0_thermal.png'  # Adjusted file name to match the format
        param_file = f'2_{row_folder.split("_")[-1]}_Parameters.txt'  # Adjusted file name to match the format
        # Copy ground truth and parameters file if they exist
        gt_image_path = os.path.join(row_path, gt_image)
        param_file_path = os.path.join(row_path, param_file)
        if os.path.exists(gt_image_path):
            shutil.copy(gt_image_path, os.path.join(training_data_folder, gt_image))
        else:
            print(f"Ground truth file not found: {gt_image_path}")

        if os.path.exists(param_file_path):
            shutil.copy(param_file_path, os.path.join(training_data_folder, param_file))
        else:
            print(f"Parameters file not found: {param_file_path}")

print("Integral imaging process complete for all rows.")