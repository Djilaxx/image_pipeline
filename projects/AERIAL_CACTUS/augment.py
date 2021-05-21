import torchvision.transforms as transforms
#The mean and std I use are the values from the ImageNet dataset
#The augmentations are used to make training harder and more robust to novel situations.
#We don't use augment on the validation set other than normalization to try and estimate the real power of the model in the wild.
Augmentations = {
    'train':
        transforms.Compose(
            [
                transforms.RandomRotation(degrees=0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]
        ),
    'valid':
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]
        )
}
