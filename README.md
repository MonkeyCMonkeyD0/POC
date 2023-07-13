# POC-Project

POC-Project is the repository of the pytorch application that what used for every research on the POC project.

## Installation

Use the package manager [conda](https://conda.io) to install `POC-env` which contain every library needed to run the application.

```bash
conda env create -f environment.yml
```

You might also need jupyter to run the application (in the base environement).
```bash
conda install jupyter
```

## Project Arechitecture

* `data/` &rarr; Folder containing the datasets ready for use by the program
	* `POC2` &rarr; Last version of the POC dataset with fixed distribution of the elements in `training` / `validation` / `testing`
	* `POCvsCS9` &rarr; Same but with masks from CS9's labeling and from POC's labeling to compare the difference made by our new labeling technique

* `src/` &rarr; Folder containing the application and its code
	* `main.ipynb` &rarr; Jupyter notebook used to run the training and testing of a model
	* `main_tune.ipynb` &rarr; Jupyter notebook used to run batches of training and testing of models for Hyperparameter Analysis
	* `dataset/ loss/ metrics/ models/ pipeline/` &rarr; Build as Python modules so can be easily imported in the notebook for execution
	* `train.py & train_tqdm.py` &rarr; Training loop (as module) with or witout the `tqdm` library for nice progress bar during training

	* `dataset/` &rarr; Each file is a different Pytorch Dataset. Composed of 2 parts: first a `DataReader()` that read the files then a `Dataset()` that load the image and apply the necessary transformations
		* `CS9_dataset.py` &rarr; For the old CS9 dataset (not in use anymore)
		* `fixed_POC_dataset.py` &rarr; For the POC2 and CS9 dataset (most recent version that should be used) with fixed distribution of the elements
		* `POC_dataset.py` &rarr; For the old POC dataset with random distribution (not in use anymore)
	* `loss/` &rarr; All loss functions used with each its own file
  		* `combination_loss.py` &rarr; Contains MeanLoss and BorderLoss
    * `metrics/` &rarr; Each metrics used for the evaluation and a Python class to record the evolution of the `training` / `validation` / `testing`
	* `model/` &rarr; Each network is its own file
 	* `pipeline/` &rarr; The input pipeline used to edit the images before feeding them to the network
  		* `filters/` &rarr; All the available filters for the pipeline in splited in 3 categories:
			* `basics` &rarr; Basics filter that can be used direcly or in combination with others
   			* `small/medium/large kernel` &rarr; Filter with different kernel size (Can easily switch kernel size in the library import)
    		* `work_in_progress` &rarr; Every other filter that we did not used or that doesn't work
