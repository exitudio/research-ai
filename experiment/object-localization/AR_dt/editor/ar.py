import math

import cv2
from keras.applications.mobilenet import MobileNet, _depthwise_conv_block
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence
import numpy as np

num = np.array([[1,2],[3,4]])
print( np.multiply(num, 4) )

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
model.load_weights('model-0.96.h5')

cap = cv2.VideoCapture(0)
while(1):

  ret, raw_frame = cap.read()
  if not ret:
      break

  # crop
  h = raw_frame.shape[0]
  w = raw_frame.shape[1]
  start_x = int((w - h) / 2)
  display_frame = raw_frame[:, start_x: start_x+h]
  # resize
  predict_frame = cv2.resize(display_frame, (IMAGE_SIZE, IMAGE_SIZE))

  points = model.predict(x=np.array([predict_frame]))[0].astype(int)
  points = np.array([[points[0], points[1]], 
                      [points[2], points[3]], 
                      [points[6], points[7]], 
                      [points[4], points[5]]])
  scale = (720/IMAGE_SIZE)
  points = np.multiply(points, scale).astype(int)

  cv2.polylines(display_frame, [points], True, (255,255,255))
  cv2.imshow('frame',display_frame)
  k = cv2.waitKey(5) & 0xFF
  if k == 27:
      break

cv2.destroyAllWindows()

