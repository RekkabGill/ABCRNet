# ABCRNet

ABCRNet is a CNN developed to predict compartment membership ratios using a reference genome (i.e., DNA sequence) as input. This project is an extensive modification of ABCNet originally developed by Matthew Kirchhoff.

M. Kirchhof, C. J. Cameron and S. C. Kremer, "End-to-end chromosomal compartment prediction from reference genomes," 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2021, pp. 50-57, doi: 10.1109/BIBM52615.2021.9669521 

## Environment

The following packages are required:

1) Python 3.9.4 or later -> Earlier versions of python3 should also work just fine
2) Numpy -> Install using pip with "pip install numpy"
3) PyTorch -> Install by visiting https://pytorch.org/get-started/locally/

## Running ABCRNet:

1. Preprocess the reference genome and prepare the training and testing data
```
DataPreprocessing.py name_of_folder_inside_Data_folder name_of_file_w_extension

positional arguments:

    name_of_folder_inside_Data_folder       the folder inside the Data folder that holds the compartment data and will hold results
    name_of_file_w_extension                the name of the compartment file with the extension (i.e. .txt)
```

2. Run the ABCNet algorithm, train your model and test it on a withehld chromosome
```
ABCModelHarness.py chromosome_usedfor_testing folder_inside_Data

positional arguments:

    name_of_folder_inside_Data_folder       the folder inside the Data folder that holds the compartment data and will hold results
    chromosome_usedfor_testing              chromosome not be used for training but instead will be witheld for testing and accuracy
```
