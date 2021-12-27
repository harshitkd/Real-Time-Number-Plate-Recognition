# Real-Time-Number-Plate-Recognition
![1_-1D-CabftTbQpm01cp3Dpg](https://user-images.githubusercontent.com/56076028/145536187-50aaa69f-655c-4459-8353-78327a2c0f7c.gif)


## Tech Stack
* [yolov4](https://github.com/theAIGuysCode/yolov4-custom-functions) : I used this OD model because it performs much better than traditional computer vision methods.
* [Easy OCR](https://github.com/JaidedAI/EasyOCR) : In this project I used EasyOCR to extract text and leverage a size filtering algorithm to grab the largest detection region. EasyOCR is build on PyTorch.
* [openCV](https://opencv.org/): It is a library mainly used at real-time computer vision.
* [Tensorflow](https://github.com/tensorflow/models) : Here I used Tensorflow object detection Model (SSD MobileNet V2 FPNLite 320x320) to detect the plate trained on a Kaggle Dataset.
* Python Libraries: Most of the libraries are mentioned in [requirements.txt](https://github.com/harshitkd/Real-Time-Number-Plate-Recognition/blob/main/requirements.txt) but some of the libraries and requirements depends on the user's machines, whether its installed or not and also the libraries for Tensorflow Object Detection (TFOD) consistently change. Eg: pycocotools, pytorch with CUDA acceleration (with or without GPU), microsoft visual c++ 19.0 etc.

## Steps
These outline the steps I used to go through in order to get up and running with ANPR. 

### Install and Setup :

<b>Step 1.</b> Clone this repository: https://github.com/harshitkd/Real-Time-Number-Plate-Recognition
<br/><br/>
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv arpysns
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source tfod/bin/activate # Linux
.\arpysns\Scripts\activate # Windows 
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=anprsys
</pre>
<br/>

### Dataset: 
Used the [Car License Plate Detection](https://www.kaggle.com/andrewmvd/car-plate-detection) kaggel dataset and manually divided the collected images into two folders train and test so that all the images and annotations will be split among these two folders.

### Training Object Detection Model
I used pre-trained state-of-the-art model and just fine tuned it on our particular specific use case.Begin the training process by opening [Real Time Number Plate Detection](https://github.com/harshitkd/Real-Time-Number-Plate-Recognition/blob/main/Real%20Time%20Number%20Plate%20Detection.ipynb) and installed the Tensoflow Object Detection (TFOD) 

![68747470733a2f2f692e696d6775722e636f6d2f465351466f31362e706e67](https://user-images.githubusercontent.com/56076028/145552503-b3a442a4-03bf-467e-af74-3e218c949dad.png)

In the below image you will see the object detection model which is now trained. I have decided to train it on the terminal because the training inside a separate terminal on a windows machine displays live loss metrics.

![Screenshot (72)](https://user-images.githubusercontent.com/56076028/145536355-94f60307-3632-4bd4-9eb7-02b9c875471d.png)

* Visualization of Loss Metric, learning rate and number of steps:

<pre>
tensorboard --logdir=.
</pre>

![tensorboard loss](https://user-images.githubusercontent.com/56076028/145684910-d237be53-88d4-45fa-b36e-dd9a52daf8e1.jpg)

![tensorboard learning and steps](https://user-images.githubusercontent.com/56076028/145684923-36a95279-5b27-4f25-bd2d-ea58eaa82075.jpg)

### Detecting License Plates

![Screenshot 2021-12-10 130124](https://user-images.githubusercontent.com/56076028/145536393-986af131-ce84-4d4c-8174-735ed492a45b.jpg)


### Apply OCR to text

<pre>
import easyocr
detection_threshold=0.7
image = image_np_with_detections
scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]
</pre>

![Screenshot 2021-12-10 125508](https://user-images.githubusercontent.com/56076028/145536427-d27c0fdc-cd30-446b-9b16-6408fdb4efcd.jpg)

### Results

Used this in real time to detect the license plate and stored the text in .csv file and images in the Detection_Images folder.

### Object Detection Metric:
![evaluation metric](https://user-images.githubusercontent.com/56076028/145684944-29306983-8396-47a2-9a08-f13a86d56f08.jpg)

![evaluation metric detail](https://user-images.githubusercontent.com/56076028/145684945-7f17e0b6-e623-4a71-b163-388a84d713fd.jpg)

<pre>
tensorboard --logdir=.
</pre>

![mAP](https://user-images.githubusercontent.com/56076028/145684953-51fc55d3-c9cd-4789-807e-0cfa0196000c.jpg)

![AR](https://user-images.githubusercontent.com/56076028/145684962-3236958f-4354-4230-b8d2-c59d18665b31.jpg)

#### For all other details and errors I faced, you can see [here](https://github.com/harshitkd/Real-Time-Number-Plate-Recognition/wiki/Details-and-Errors)
