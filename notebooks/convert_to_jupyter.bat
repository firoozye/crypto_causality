
@echo off
# This script converts Python scripts in the 'python' directory to Jupyter notebooks in the 'jupyter' directory.
# 1 The jupyter command is available in your PATH.
# 2 You have the nbconvert package installed (pip install nbconvert).
# 3 The script is run from the notebooks directory containing 'python' and 'jupyter' subdirectories.

# Check if the jupyter command is available
where jupyter >nul 2>nul
if %errorlevel% neq 0 (
    echo jupyter command not found. Please ensure it is installed and available in your PATH.
    exit /b 1
)

# Check if the nbconvert package is installed
pip show nbconvert >nul 2>nul
if %errorlevel% neq 0 (
    echo nbconvert package not found. Please install it using pip install nbconvert.
    exit /b 1
)
# Check if the python directory exists
if not exist "python" (
    echo The 'python' directory does not exist. Please create it and place your Python scripts inside.
    exit /b 1
)
# Check if the jupyter directory exists, if not create it
if not exist "jupyter" (
    mkdir jupyter
    if %errorlevel% neq 0 (
        echo Failed to create the 'jupyter' directory. Please check your permissions.
        exit /b 1
    )
)


 @echo off

 REM Convert Python scripts to Jupyter notebooks
 for %%f in (python\*.py) do (
     jupyter nbconvert --to notebook --output-dir=jupyter --output="%%~nf.ipynb" "%%f"
 )

 echo Conversion complete. Jupyter notebooks are now in the 'jupyter' directory.
