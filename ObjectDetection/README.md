# Object Detection with Drinks Dataset using Faster R-CNN
In this experiment, we finetuned an object detection model for specific types of drinks. 

### Model reference
The following implementation makes use of Faster R-CNN to perform object detection. To know more about the model, you may check out the links to the paper and code:
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
* [Faster R-CNN Python Implementation](https://github.com/rbgirshick/py-faster-rcnn)

### Data preparation
This object detection model makes use of the drinks dataset. You may download the dataset together with the labels from [https://bit.ly/adl2-ssd](https://bit.ly/adl2-ssd). However, you do not need to download this if you will be running ```train.py``` or ```test.py``` since the scripts will automatically prepare the data for you if the dataset is still not present in the path.

### Training the model
To train the model on the train dataset and test its performance on the test dataset,
```
python train.py
```
The script will automatically save the newly trained model weights to the filename ```drinks_object-detection.pth```.

### Testing the model
To test the model on the test dataset with the pre-trained model weights from ```drinks_object-detection.pth```,
```
python test.py
```
The script will automatically download the pre-trained model weights (even without training) and load it to the model.

### Object detection demo
To perform a real-time demo of the project,
```
python demo.py
```
It is recommended to have CUDA enabled for more real-time inference. Running inference on CPU may result in slower prediction speed than the frame rate.\
To exit the demo, press 'q' on your keyboard.

### Other references
Here are other references that were used to implement the object detection program:
* [Torchvision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
* [PyTorch Vision Detection Reference](https://github.com/pytorch/vision/tree/main/references/detection)
* [Deep Learning Experiments: Datasets & Dataloaders](https://github.com/roatienza/Deep-Learning-Experiments/tree/master/versions/2022/datasets/python)
* [Demo Reference Code](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter11-detection/video_demo.py)
