import cv2
import csv
import os
import numpy as np

PATH = 'data'
CSV_PATH = f'{PATH}/corners.csv'

STATE_CAPTURE = 'state_capture'
STATE_CLICK = 'state_click'
WINDOW_NAME = 'editor'
MAX_CLICK = 4
state = {
  'step': STATE_CAPTURE,
  'num_click': 0,
  'id': 0,
  'data': []
}
raw_frame = None
display_frame = None

def run_cam():
  global display_frame, raw_frame
  while True:
    if state['step'] == STATE_CAPTURE:
      ret, raw_frame = cam.read()
      if not ret:
          break
      
      # crop
      h = raw_frame.shape[0]
      w = raw_frame.shape[1]
      start_x = int((w - h) / 2)
      display_frame = raw_frame[:, start_x: start_x+h]
      # resize
      raw_frame = cv2.resize(display_frame, (224, 224))

    cv2.imshow(WINDOW_NAME, display_frame)
    k = cv2.waitKey(100)

def clickPhase(event, x, y, d, e):
  global display_frame, raw_frame
  if event == cv2.EVENT_LBUTTONUP:
    if state['num_click'] == 0:
      state['step'] = STATE_CLICK
      state['data'].append([])
    elif state['num_click'] >= 1 and state['num_click'] <= MAX_CLICK:
      cv2.circle(display_frame, (x, y), 2, (0,0,255), -1)
      state['data'][state['id']].append(x)
      state['data'][state['id']].append(y)
    elif state['num_click'] > MAX_CLICK:

      # scale down to 244
      scale = (224/720)
      state['data'][state['id']] = np.multiply(state['data'][state['id']], scale).astype(int)


      # show rectangle
      # one_image_corners = state['data'][state['id']]
      # resequence_corner = np.int32([ (one_image_corners[0], one_image_corners[1]),
      #                                (one_image_corners[2], one_image_corners[3]),
      #                                (one_image_corners[6], one_image_corners[7]),
      #                                (one_image_corners[4], one_image_corners[5])] )
      # cv2.polylines(raw_frame, [resequence_corner], True, color=(0, 0, 255, 255), thickness=1)

      # save image
      img_name = f'{PATH}/image_{state["id"]}.png'
      cv2.imwrite(img_name, raw_frame)
      
      # save csv
      with open(CSV_PATH, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(state['data'])

      # set state
      state['id'] += 1
      state['num_click'] = -1
      state['step'] = STATE_CAPTURE
    state['num_click'] += 1

def load_csv():
  if os.path.isfile(CSV_PATH):
    with open(CSV_PATH, 'r') as f:
      reader = csv.reader(f)
      state['data'] = list(reader)
      state['id'] = len(state['data'])

cam = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, clickPhase)

load_csv()
run_cam()

cam.release()
cv2.destroyAllWindows()




