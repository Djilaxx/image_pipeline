# **Image Pipeline**

This repository contains an image machine learning pipeline and some of the projects i've worked on, either for fun/education or competition on Kaggle. \
Each project have it's own readme containing information about the specific problematics of each. 

I train the models locally on my pc using a Nvidia 1080 GPU. 
## **Data**

The data is not in the repository directly if you want to launch a model on one the projects in here you must download the data and change the config file in the task folder to be adequate. \
Links to the datasets are in the tasks README's.
## **Projects**
---
The projects folder contains the specific code about each project :
 * config.py file containing  most of the parameters necessary to train a model.
 * augment.py that contain the specific augmentations you want to perform on the images for training, validation and testing

### **What if i want to add a new project ?**
To add a new project you'll to create a few things : 
* a new folder in the projects/ folder containing a \_\_init\_\_.py file, a config.py file and a augment.py file
* copy and paste the content of another config.py file and change the information to be adequate with your task
* Add the augment you require to the augment.py file

### **Training**

To start training a model on any task use this command in terminal :
```
python -m train --project=AERIAL_CACTUS --run_note=test
```
You can replace the **AERIAL_CACTUS** with any folder in projects/.
Default parameters train a **RESNET18** model.
You can change these parameters as such :
```
python -m train --project=LEAF_DISEASE --model_name=RESNET34 --run_note=test
```

The parameters can take different values :
*- **run_note** : str, allow you to add a title to your training runs
* **project** : The project you want to train a model on, atm you can train a model on the aerial_cactus task, melanoma, blindness_detection & leaf_disease projects.
* **model_name** : You can choose any model that is in the models/ folder, name must be typed in MAJ like in the example above.

### **Inference**
To start prediction on new data for a project you can use this :
```
python -m predict --project=AERIAL_CACTUS --model_name=RESNET18 --run_note=test
```

## **To do** 
---
* Add more models
* Add more loss functions available
* Add notebooks (for model evaluation - EDA - hyperparameter optimization etc...)
* add support for meta features