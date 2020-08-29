# About face-recognition
This is a Face recognition system using Opencv.
This system is very accurate to recognise faces. It will be more accurate if you train this model on more face data.
I've used two types of face data here.. One is Shah rukh khan's face data and the 2nd one is Thomas Shelby's(Main character from Peaky Blinders) face data.
I've taken 20 pictures of each people.

we are detecting a human face using "haarcascade_frontalface_default" file.
Capturing the video with Opencv

# Requirements

```
import os
from PIL import Image
import numpy as np
import cv2
import sys
import pickle
```
