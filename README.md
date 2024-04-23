    
# Object Detection and Segmentation

This project implements a system for segmentation and detection of abnormal clods that can form during the processing of rock in the coal industry. In order to make sure that the mechanism that processes the breed is working correctly.

Here you can see a fragment of the video with tracking processed by the system:
!(./outputs/2024-04-21 19-14-53.gif)

## Models

The project uses two types of models: [Ultralytics YOLOv8](https://ultralytics.com/yolov8 ) (fully operational with tracking) and [U-Net](the model was implemented from scratch)

##  Installation
 Run following command in shell:\
``` 
$ pip install -r requirements.txt
```

## User Gide
### CLI

The system works through the following CLI commands:
1. train:
```$ python path/to/model.py train --parameter=value```

2. evaluate: ```$ python path/to/model.py evaluate --parameter=value```


4. demo: ```$ python path/to/model.py demo --parameter=value```

``` 
Parameters:
    tracking: bool; by default: False. If true, the function includes object tracking in the real-time detection.
Output: return frames in live
```
