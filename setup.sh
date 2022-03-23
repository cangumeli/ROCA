# conda environment
conda create -n roca python=3.8
conda activate roca

# Simple deps
pip install -r requirements.txt

# pytorch3d
conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
# NOTE: Tested with 0.5 and 0.6
conda install pytorch3d -c pytorch3d-nightly

# detectron2
python -m pip install detectron2==0.3 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

# numba
conda install numba
