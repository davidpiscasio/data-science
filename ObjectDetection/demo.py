import os
import cv2
import label_utils
import torch
import gdown

from torchvision import transforms

from utils import get_model_instance_segmentation

class  VideoDemo():
    def __init__(self,
                 detector,
                 camera=0,
                 width=640,
                 height=480,
                 record=False,
                 filename="demo.mp4",
                 device=torch.device('cpu')):
        self.camera = camera
        self.detector = detector
        self.width = width
        self.height = height
        self.record = record
        self.filename = filename
        self.videowriter = None
        self.device = device
        self.initialize()

    def initialize(self):
        self.capture = cv2.VideoCapture(self.camera)
        if not self.capture.isOpened():
            print("Error opening video camera")
            return

        # cap.set(cv2.CAP_PROP_FPS, 5)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.record:
            self.videowriter = cv2.VideoWriter(self.filename,
                                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                10,
                                                (self.width, self.height), 
                                                isColor=True)

    def loop(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        line_type = 1

        while True:
            _, image = self.capture.read()

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = transforms.ToTensor()(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)

            pred = self.detector(img)
            rects = pred[0]['boxes'].cpu().tolist()
            class_idxs = pred[0]['labels'].cpu().tolist()
            scores = pred[0]['scores'].cpu().tolist()

            items = {}
            for i in range(len(class_idxs)):
                rect = rects[i]
                x1 = rect[0]
                y1 = rect[1]
                x2 = rect[2]
                y2 = rect[3]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                index = class_idxs[i]
                name = label_utils.index2class(index) + f' | {scores[i]*100:.2f}%'
                color = label_utils.get_box_rgbcolor(index)
                if name in items.keys():
                    items[name] += 1
                else:
                    items[name] = 1
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

                cv2.putText(image,
                            name,
                            (x1, y1-15),
                            font,
                            0.5,
                            color,
                            line_type)

            cv2.imshow('image', image)
            if self.videowriter is not None:
                if self.videowriter.isOpened():
                    self.videowriter.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes=4)

    if not os.path.exists('drinks_object-detection.pth'):
        url = 'https://drive.google.com/uc?id=1YlfXC1R2rH7pBb7jbxhk61XIW4QJVRnB'
        output = 'drinks_object-detection.pth'
        gdown.download(url, output, quiet=False)
        print("Downloaded pretrained model weights: drinks_object-detection.pth")

    if torch.cuda.is_available():
        model.load_state_dict(torch.load('drinks_object-detection.pth'))
    else:
        model.load_state_dict(torch.load('drinks_object-detection.pth', map_location=torch.device('cpu')))
    
    model.to(device)
    model.eval()

    videodemo = VideoDemo(detector=model,
                            camera=0,
                            record=False,
                            device=device)
    videodemo.loop()