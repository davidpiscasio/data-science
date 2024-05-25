import os
import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
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
train_dict, train_classes = label_utils.build_label_dictionary("drinks/labels_train.csv")


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 4
    # use our dataset and defined transformations
    dataset = DrinksDataset(train_dict, transforms.ToTensor())
    dataset_test = DrinksDataset(test_dict, transforms.ToTensor())

    # define training and test data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = utils.get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=30,
                                                   gamma=0.1)

    # training for 70 epochs
    num_epochs = 70

    for epoch in range(num_epochs):
        # train for one epoch, printing every 125 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=125)
        # update the learning rate
        lr_scheduler.step()

    # after training, evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    # save the model weights
    torch.save(model.state_dict(), 'drinks_object-detection.pth')

    print("Saved model weights to drinks_object-detection.pth")
    
if __name__ == "__main__":
    main()
