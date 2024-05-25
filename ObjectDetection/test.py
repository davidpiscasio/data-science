import os
import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import gdown
from engine import evaluate
import utils
from torchvision import transforms
import label_utils
from data import download_dataset, DrinksDataset

# download dataset if it doesn't exist
if not os.path.exists('drinks'):
    download_dataset()
else:
    print("Drinks dataset already in path")

test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 4
    # use our dataset and defined transformations
    dataset_test = DrinksDataset(test_dict, transforms.ToTensor())

    # define test data loaders
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = utils.get_model_instance_segmentation(num_classes)

    # download pretrained model weights
    if not os.path.exists('drinks_object-detection.pth'):
        url = 'https://drive.google.com/uc?id=1YlfXC1R2rH7pBb7jbxhk61XIW4QJVRnB'
        output = 'drinks_object-detection.pth'
        gdown.download(url, output, quiet=False)
        print("Downloaded pretrained model weights: drinks_object-detection.pth")
    else:
        print("Pretrained model already in path")

    # load the pretrained model weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('drinks_object-detection.pth'))
    else:
        model.load_state_dict(torch.load('drinks_object-detection.pth', map_location=torch.device('cpu')))

    # move model to the right device
    model.to(device)

    # evaluate the model on the test dataset
    evaluate(model, data_loader_test, device=device)
    
if __name__ == "__main__":
    main()
