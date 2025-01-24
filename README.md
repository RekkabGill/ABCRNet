# ABCRNet

ABCRNet is a CNN developed to predict compartment membership ratios using a reference genome (i.e., DNA sequence) as input. This project is an extensive modification of ABCNet[1-2].

[1] Kirchhof, Matthew. "ABCNet: Predicting Chromosomal Compartments Directly from Reference Genomes." University of Guelph, 2021.

[2] M. Kirchhof, C. J. Cameron and S. C. Kremer, "End-to-end chromosomal compartment prediction from reference genomes," 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2021, pp. 50-57, doi: 10.1109/BIBM52615.2021.9669521 

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

## CMR Analysis Script:
The CMR Analysis script contains various functions that can help analyize ABCRNet results (e.g., training and testing loss, predictions, GC-content, statistical correlations, etc.) and generate various plots that can help visualize the model output.

### Running CMR_Analysis.py

1. Under \_\_main\_\_ change the "#GLOBAL PATHS" so that the data paths are correct for each function that you want to use

2. To use the script, uncomment function of choice under "#MAIN METHODS" in \_\_main\_\_

```
CMR_Analysis.py species

positional arguments:

    species    can be either "human" or "mouse"
```

## ABCRLin:

ABCRLin is a linear model used to compare to ABCRNet. The input is the amount of GC-content from 250 kb bins. The output is the target CMR values. The model uses the gc_content.csv in the data_files. 

### Running ABCRLin.py

1. To run, adjust the data input/output paths in the file to point to the correct locations
```
ABCRLin.py species

positional arguments:

    species    can be either "human" or "mouse"
```


