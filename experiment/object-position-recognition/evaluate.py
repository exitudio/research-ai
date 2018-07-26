from train_model import create_model, IMAGE_SIZE, ALPHA
from PIL import Image, ImageDraw
import glob
import csv
import cv2
import pathlib
import numpy as np

TEST_SET_FOLDER = './data/circles_test'
OUTPUT_TEST_PATH = './data/outpute_test'
def main():
  model = create_model(IMAGE_SIZE, ALPHA)
  model.load_weights('model-0.98.h5')

  image_paths = sorted(glob.glob('{}/*png'.format(TEST_SET_FOLDER)))
  for i, image_path in enumerate(image_paths):
    image = cv2.resize(cv2.imread(image_path), (IMAGE_SIZE, IMAGE_SIZE))
    x, y, w, h = model.predict(x=np.array([image]))[0]
    cv2.rectangle(image,(int(x-w/2) , int(y-h/2)), (int(x+w/2), int(y+h/2)),(0,0, 255),1)
    pathlib.Path(OUTPUT_TEST_PATH).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f'{OUTPUT_TEST_PATH}/image{i}.png', image)
if __name__ == '__main__':
  main()