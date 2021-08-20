#!/bin/bash
# A script for installing environments

# We don't create a virtual environment, because there was an installation issue for 
# PyPDF4(the code wouldn't work in the virtual environment) 

# A step before running this file: go to cssuing home directory and copy the source to 
# you own home directory using this command

    # cp -r ../cssuing/copy_source_here ~

# We install two requirements.txt files because some libraries only work with conda installation
SOURCEDIR_pip=/home/cssuing/copy_source_here/requirements.txt
SOURCEDIR_conda=/home/cssuing/copy_source_here/conda_requirements.txt

# install this conda_requirements.txt to create the working environment that is the same as cssuing home directory
conda install --file $SOURCEDIR_conda

# install this requirements.txt to create the working environment that is the same as cssuing home directory
pip install -r $SOURCEDIR_pip
