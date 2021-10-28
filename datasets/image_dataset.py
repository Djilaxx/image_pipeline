#Data functions, allow to load the data and targets, transform into a pytorch dataset
import torch
import cv2
import albumentations as A

class IMAGE_DATASET:
    '''
    Pytorch class to define an image dataset
    image_path : must be a list of path to individual images like "data/image_001.png"
    resize : if not None, image will be resized to this size, MUST BE A TUPLE
    label : labels for each image of their class
    transforms : if not None, transform will be applied on images
    '''

    def __init__(self, image_path, resize=None, label=None, transforms=None):
        self.image_path = image_path
        self.resize = resize
        self.label = label
        self.transforms = transforms

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
            if self.resize is not None:
                A.Compose(
                    [
                        self.transforms,
                        A.Resize(self.resize[0], self.resize[1], p=1)
                    ],
                    p=1.0,
                )
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


