# Bacterial viability identification

Code and data contained in this folder were created for the publication, 
"**Classification between Live and Dead Foodborne Bacteria using Hyperspectral Microscopy Imagery and Machine Learning.**" 

Please note that the scripts and data are presented AS IS, and without any warranty or guarantee. 
And we have made a concerted effort to properly attribute credit for the code in our project. 
However, if we have missed giving credit to anyone for their contributions, we sincerely 
apologize. If you come across any missing records, please do not hesitate to reach out to 
taesung.shin@usda.gov, and we will promptly correct the oversight. 

### Data
The "data" subfolder contains both the average spectra (stored in "single_cells_scinet.csv") and 546nm band images (stored in "single_cells/*.tif") of individual cells. These cells were extracted from hyperspectral images that were captured using a dark-field hyperspectral microscope, with a wavelength range of 400-1000nm.

### Code
Python scripts were developed using Python 3.7 to train and evaluate the three Fusion-Net models (I, II, and III) introduced in the paper for detecting the viability of foodborne bacterial cells.

### Instruction to run the scripts

The following steps describe how to run the scripts:
1. Create a folder and extract all data and scripts from the provided zip file in the folder.
2. Create a new subfolder "results" in the folder made in Step 1. 
Results of training and evaluation of the models will be stored in this subfolder.
3. Run following commands in the folder.
```
virtualenv venv
venv\scripts\activate
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

Please note that the provided Python scripts assume that your machine has a GPU with 8GB or more VRAM and other necessary configurations in order to properly run tensorflow-backed Keras scripts under Python v3.7. It is important to note that these scripts may run very slowly, depending on the configuration of your machine.

### Instruction for results of training and evaluation of a model
After each model training, the Python scripts create a "fusion-netx_xxx" subfolder in the "results" folder. This subfolder contains everything that is necessary to evaluate the trained model and optimize hyperparameters, including the trained model (stored in the "model" subfolder), data scaler (*_scaler.bin), and an index of the training, validation, and test data in the original dataset (stored in "index.mat").

The simplest way to check the evaluation results of the model may be to read the "summary.txt" file. In this text file, locate the line that reads "Evaluation of best performing model" and look for the subsequent lines that indicate the loss and accuracy values of the model, including for the training, validation, and test data.

For example, let's assume you have a summary.txt file containing following lines
```
Evalutation of best performing model:
[0.0410853862762451, 0.9418933629989624]
[0.0471896171569824, 0.9337423086166382]
[0.0587867140769958, 0.915625]
```
Based on the content, the loss and accuracy of the model were 0.041 and 94% with the training data, 0.047 and 93% with the validation data, and 0.059 and 92% with the test data.

However, it is important to note that some sections of the results may not be directly related to model evaluation since the scripts were written for both model training and hyperparameter optimization.

The provided scripts and data have been validated with several computing environments (e.g., PC with Windows 10 and cloud with Linux), but if you encounter any issues, please send an email to taesung.shin@usda.gov. Thank you.
