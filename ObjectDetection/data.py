from genericpath import exists
from PIL import Image

import gdown
import tarfile
import torch

def download_dataset():
    url = 'https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'
    output = 'drinks.tar.gz'
    gdown.download(url, output, quiet=False)
    file = tarfile.open('drinks.tar.gz')
    file.extractall('./')
    file.close()
    print("Finished downloading drinks dataset")

class DrinksDataset(object):
    def __init__(self, dictionary, transform=None):
        self.dictionary = dictionary
        self.transform = transform

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # retrieve the image filename
        key = list(self.dictionary.keys())[idx]
        # retrieve all bounding boxes
        boxes = self.dictionary[key]
        # swap xmax and ymin to conform to appropriate format
        boxes[:,1:3] = boxes[:,2:0:-1]
        # retrieve labels
        labels = boxes[:,4]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # remove label from bounding boxes
        boxes = boxes[:,0:4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        # open the file as a PIL image
        img = Image.open(key)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # apply the necessary transforms
        if self.transform:
            img = self.transform(img)
        
        # return a list of images and target (as required by the Faster R-CNN model)
        return img, target