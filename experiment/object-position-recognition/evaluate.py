from train_model import create_model, IMAGE_SIZE, ALPHA
import glob
import csv

TEST_SET_FOLDER = './data/circles_test'
def main():
  # model = create_model(IMAGE_SIZE, ALPHA)
  # model.load_weights('model-0.95.h5')

  # images = sorted(glob.glob('{}/*png'.format(TEST_SET_FOLDER)))

  with open(f'{TEST_SET_FOLDER}/locations.csv', "r") as file:
    reader = csv.reader(file, delimiter=",")
    arr = list(reader)
  print('arr:', arr)
if __name__ == '__main__':
  main()