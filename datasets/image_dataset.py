#Data functions, allow to load the data and targets, transform into a pytorch dataset
import torch
import cv2
import numpy as np 
import albumentations as A

class IMAGE_DATASET:
    """
    Torch-style dataset for image classification or regression tasks

    Parameters:
    -----------
    image_path: list
        list of path to images like: data/project_folder/image_id001.jpg
    resize: tuple
        tuple of integer giving the height and widthfor the resize: (width, height)
    label: list or None
        list of labels indicating the class to predict for a given image - None if using test images
    transforms: albumentation Compose object
        albumentation list of transforms you wish to apply to your images

    Returns:
    --------
    dictionnary including images and labels as tensors ready for training
    or just images if labels were not included
    """
    
    def __init__(self, image_path, resize=None, label=None, transforms=None):
        self.image_path = image_path
        self.resize = resize
        self.label = label
        self.transforms = transforms

        if self.resize is not None:
            self.transforms = A.Compose(
                [
                    A.Resize(self.resize[0], self.resize[1], p=1),
                    self.transforms
                ],
                p=1.0,
            )

    #RETURN THE LENGHT OF THE DATASET
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        #LOADING IMAGES
        image = cv2.imread(self.image_path[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #APPLYING DATA AUGMENTATIONS TO IMAGE DATA
        if self.transforms is not None:
            #Adding resize transforms to the augmentation list if provided
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.label is not None:
            label = self.label[item]
            return {
                "images": image, 
                "labels": torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                "images": image
            }


