#!/bin/bash
# This script converts Python scripts in the 'python' directory to Jupyter notebooks in the 'jupyter' directory.
# 1 The jupyter command is available in your PATH.
# 2 You have the nbconvert package installed (pip install nbconvert).
# 3 The script is run from the notebooks directory containing 'python' and 'jupyter' subdirectories.

 # Check if the jupyter directory exists, if not create it
 #
 
# check if the jupyter command is available
if ! command -v jupyter &> /dev/null
 then
     echo "jupyter command not found. Please install Jupyter Notebook."
     exit 1
 fi

 # Check if the nbconvert package is installed
 
if ! python -c "import nbconvert" &> /dev/null
 then
     echo "nbconvert package not found. Please install it using 'pip install nbconvert'."
     exit 1
 fi

# Check if the jupyter directory exists, if not create it
 if [ ! -d "jupyter" ]; then
     mkdir jupyter
 fi

 # Check if the python directory exists, if not exit
 if [ ! -d "python" ]; then
     echo "The 'python' directory does not exist. Please create it and add your Python scripts."
     exit 1
 fi

# Convert Python scripts to Jupyter notebooks
for py_file in python/*.py; do
    base_name=$(basename "$py_file" .py)
    # Use jupytext to convert properly
    if command -v jupytext &> /dev/null; then
        jupytext --to notebook "$py_file" --output "jupyter/${base_name}.ipynb"
    else
        # Fallback to nbconvert if jupytext is not available
        jupyter nbconvert --to notebook --output-dir=jupyter --output="${base_name}.ipynb" "$py_file"
    fi
done

 echo "Conversion complete. Jupyter notebooks are now in the 'jupyter' directory."



