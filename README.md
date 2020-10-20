# COMP472_A1
## Team Try Hard
Repo: https://github.com/MackTr/COMP472_A1  

Mackenzie Trenholm - 40057679  
Alessandro Sartor - 40042866  
## Install dependencies
  You need to install:  
  Numpy -> pip install numpy  
  Sklearn -> pip install -U scikit-learn  
  MatPlotLib -> python -m pip install -U matplotlib  
## Instructions to plot distribution  
 1. At the top of the file, you must write the path of your dataset files.
 2. In distribution.py at line 39, you have to specify the dataset you want to plot and the file with the indexes. For exemple, v2 is the path of the validation_dataset2.csv and   i2 is the path to index_dataset2.csv
 3. Run distribution.py to plot the distribution graph of the dataset you selected.
## Instruction to run models
1. Inside global_variables.py, write the paths to the dataset files that you want to use. Right now, the paths for dataset 1 and dataset 2 are in the file, you simply need to uncomment the right variables according to your OS.
2. Inside driver.py at line 35, simply comment out the models that you don't want to run.
3. If you don't want to write a csv file, go in the model python file (e.g. gnd.py) and turn variable "Print" to false.
4. You can also modify the name of the csv file in the model python file.
5. Run driver.py


