#Data functions, allow to load the data and targets, transform into a pytorch dataset
import torch
from PIL import Image
from PIL import ImageFile

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
        image = Image.open(self.image_path[item])

        #RESIZING IMAGES
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        #APPLYING DATA AUGMENTATIONS TO IMAGE DATA
        if self.transforms:
            image = self.transforms(image)

        if self.label is not None:
            label = self.label[item]
            return {"images": image, "labels": torch.tensor(label)}
        else:
            return {"images": image}


