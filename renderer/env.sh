# A shared data folder is assumed for all datasets
# You can modify the vatiables 
# so different things can be stored in different places
export BASE_DIR=$HOME/Data

# ScanNet25k data downloadable as a zip using the official script
export SCANNET25K_DIR=$BASE_DIR/ScanNet25k

# Scan2CAD data consisting of few json files, contact the authors to obtain
export S2C_ROOT=$BASE_DIR/Scan2CAD

# The resized images will be stored here and used as the input of everything
export IMAGE_DIR=$BASE_DIR/Images

# Rendered depths and instance segmentation labes will be saved here
export RENDERING_DIR=$BASE_DIR/Rendering

# Output dataset dir for everything except rendered images
export DATA_DIR=$BASE_DIR/Dataset

# Downloaded ShapeNet location
export SHAPENET_DIR=$BASE_DIR/ShapeNetCore.v2

export NUM_WORKERS=8  # multi-processing is very important for the speeds of resize and render scripts
export OMP_NUM_THREADS=1  # OMP is useless since we use multi-processing
