import math

import cv2
from keras.applications.mobilenet import MobileNet, _depthwise_conv_block
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence
import numpy as np


ALPHA = 0.25
IMAGE_SIZE = 224

def create_mobile_net_model(size, alpha):
  model_net = MobileNet(input_shape=(size, size, 3), include_top=False, alpha=alpha)
  x = _depthwise_conv_block(model_net.layers[-1].output, 1024, alpha, 1, block_id=14)
  x = Conv2D(64, (3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(4, 4))(x)
  x = Conv2D(8, kernel_size=(1, 1), padding="same")(x)
  x = Reshape((8,))(x)
  return Model(inputs=model_net.input, outputs=x)
model = create_mobile_net_model(IMAGE_SIZE, ALPHA)
model.load_weights('model-0.92.h5')

cap = cv2.VideoCapture(0)
while(1):

    # Take each frame
    _, image = cap.read()
    (w,h,_) = image.shape

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    points = model.predict(x=np.array([image]))[0].astype(int)
    points = np.array([[points[0], points[1]], 
                       [points[2], points[3]], 
                       [points[6], points[7]], 
                       [points[4], points[5]]])
    cv2.polylines(image, [points], True, (255,255,255))

    image = cv2.resize(image, (h, w))
    cv2.imshow('frame',image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

