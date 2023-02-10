# Bacterial viability identification

Code and data contained in this folder were created for the publication, 
"Classification between Live and Dead Foodborne Bacteria using Hyperspectral Microscopy Imagery and Machine Learning." 

Please note that the code is presented AS IS, and without any warranty or guarantee. 
Furthermore, as of February 9th, 2023, we must acknowledge that the scripts may not be functional 
as they have been extracted from HMI deep learning repository in high-performance computing clusters 
of the USDA SCINet without any modifications due to time constraints. 
Current missing parts are referencing data (in **data** folder) in the code and 
a column in CSV file needs to be changed to corresponding folder paths (./single_cells).
However, we will ensure runnable scripts in the near future as time permits.

We have made a concerted effort to properly attribute credit for the code in our project. 
However, if we have missed giving credit to anyone for their contributions, we sincerely 
apologize. If you come across any missing records, please do not hesitate to reach out to 
taesung.shin@usda.gov, and we will promptly correct the oversight. 
Thank you for your understanding.

### Instruction to run:

1. Create a folder and put all data and scripts in the folder
2. Edit scripts and data to run the scripts in the folder. (**This part is currently missing**)
3. Run following commands in the folder.
```
virtualenv venv
pip install -r requirements.txt
```
Once it's done, you can train and evaluate Fusion-Net I in the paper as follows
```
python fusion-net_1or2.py 0 0
```
The first argument is for bacterial strain filtering for training. You can use 0 for training with all strains. The second argument is for selecting model (Fusion-Net I for 0 and Fusion-Net II for 1)

Thus, following command will train and evaluate Fusion-Net II:
```
python fusion-net_1or2.py 0 1
```
To train Fusion-Net III
```
python fusion-net_3.py 0
```
