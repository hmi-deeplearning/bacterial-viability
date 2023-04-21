# Bacterial viability identification

Code and data contained in this folder were created for the publication, 
"**Classification between Live and Dead Foodborne Bacteria using Hyperspectral Microscopy Imagery and Machine Learning.**" 

Please note that the scripts and data are presented AS IS, and without any warranty or guarantee. 
And we have made a concerted effort to properly attribute credit for the code in our project. 
However, if we have missed giving credit to anyone for their contributions, we sincerely 
apologize. If you come across any missing records, please do not hesitate to reach out to 
taesung.shin@usda.gov, and we will promptly correct the oversight. 

### Data
**data** subfolder contains average spectra (single_cells_scinet.csv) and 546nm band images (single_cells/*.tif) of 
single cells that were extracted from hyperspectral images scanned with dark-field hyperspectral microscope of 
400-1000nm wavelength range.

### Code
Python scripts developed based on Python 3.7 were provided to train and evaluate three Fusion-Net models (I, II, and III) introduced 
for viability detection of foodborne bacterial cells in the paper above.

### Instruction to run the scripts

The following steps describe how to run the scripts:
1. Create a folder and extract all data and scripts from the provided zip file in the folder.
2. Create a new subfolder "results" in the folder made in Step 1. 
Results of training and evaluation of the models will be stored in this subfolder.
3. Run following commands in the folder.
```
virtualenv venv
pip install -r requirements.txt
```
4. (Optional) If the package installation was not successful, try the command below instead.
```
pip install tensorflow-gpu keras scikit-image scikit-learn pandas seaborn scipy livelossplot tqdm hyperopt hyperas opencv-python spectral pydicom kaggle imgaug
```

5. Once it's done, you can train and evaluate Fusion-Net I in the paper as follows
```
python fusion-net_1or2.py 0 0
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The first argument is for bacterial strain filtering for training. 
You can use 0 for training with all strains. The second argument is for selecting model (Fusion-Net I for 0 and Fusion-Net II for 1)

6. Thus, following command will train and evaluate Fusion-Net II:
```
python fusion-net_1or2.py 0 1
```
7. To train Fusion-Net III
```
python fusion-net_3.py 0
```

Please note that these scripts have been written assuming your machine has a GPU with 8GB VRAM and other configurations 
to properly run tensorflow-backed Keras scripts under Python v3.7. 
The scripts can run extremely slow, depending on the configuration of your machine. 

### Instruction for results of training and evaluation of a model
After each model training, the scripts create a **fusion-netx_xxx** subfolder in results folder. The subfolder contains everything necessary 
to evaluate your model and optimize hyperparameters, 
including model (**model** subfolder), data scaler (***_scaler.bin**), 
index of training, validation, and test data in original dataset (**index.mat**).

The simplest way to check model evaluation results may be 
reading summary.txt file. 
In the text file, 
find a line of "Evalutation of best performing model" and subsequent lines 
indicating loss and accuracy values of the model with training, validation, and test data.

For example, let's assume you have a summary.txt file containing following lines
```
Evalutation of best performing model:
[0.0410853862762451, 0.9418933629989624]
[0.0471896171569824, 0.9337423086166382]
[0.0587867140769958, 0.915625]
```
The content implies that loss and accuracy of model were 
0.041 and 94% with training data, 0.047 and 93% with validation data,
0.059 and 91% with test data.

It must be noted that some portions of the results may not be directly related to model evaluation 
because the scripts were written for both model training and hyperparameter optimizations, 

The scripts and data were validated with several computing environments but 
if you find any problem, please send an email to taesung.shin@usda.gov. Thanks.