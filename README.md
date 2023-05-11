# Project Name
Voxel Grid Optimization

# Description
This project uses Plenoxels to optimize a 3D voxel grid from 2D images for RGB values, resulting in a much lighter model to run. The optimized grid can then be used for various computer vision tasks such as object detection and segmentation.

## Installation
To install the project, follow these steps:
1. Make sure you have Python and all necessary dependencies installed.
2. Navigate to the root directory of the project.
3. Run `pip install -r requirements.txt`.
4. Run `python scripts/main.py` to start the training process.

## Scripts
The following scripts are available in the `scripts` directory:

#### compare_inference_to_image.py
Compares the output of the trained model with the input image.

#### main.py
The main script that runs the entire optimization process. It takes care of loading the dataset, preprocessing the images, initializing the model, training the model, evaluating its performance, and saving the results.

#### train.py
A script that trains the model using the provided dataset.

#### visulize_camera_and_grid.py
Visualizes the camera position and orientation along with the voxel grid.

#### visulize_grid.py
Displays a 3D view of the voxel grid.

## Dataset
The current dataset consists of two folders named 'test' and 'train'. Each folder contains subfolders named after objects (e.g., 'chair', 'table'). Inside each object folder, there are three more subfolders named 'val', 'train', and 'test'. These subfolders contain PNG images of the corresponding object from different angles and distances.

## Grid Size
By default, the grid size is set to 32x32x32. However, you can change this by modifying the `GRID_SIZE` variable in `data_processing.py`. Note that increasing the grid size may result in longer training times and larger file sizes.

## Saving Results
The trained model and other relevant information will be saved in the `src/grid_cells_trained.*` files. You can load these files later to use the already-optimized grid for inference.