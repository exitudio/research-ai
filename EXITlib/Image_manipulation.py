import math
import cv2
import numpy as np

class Image_manipulation:
  def get_rotated_point(self, rotation_matrix, points):
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    rotated_point = rotation_matrix.dot(points_ones.T).T
    return rotated_point
  
  def get_rotate_matrix(self, corners, radian_angle, pivot, offset_x=0, offset_y=0):
    scale = 1
    x, y = pivot
    a = scale*math.cos(radian_angle)
    b = scale*math.sin(radian_angle)
    M = np.float32([[a, b, (1-a)*x - b*y + offset_x],
                    [-b, a, b*x + (1-a)*y + offset_y]])
    rotated_points = self.get_rotated_point(M, corners)
    rotated_points = np.array(rotated_points)
    min_point = np.min(rotated_points, 0)
    max_point = np.max(rotated_points, 0)
    return M, min_point, max_point

    
  def rotate_bound(self, image, angle, corners=None):
      radian_angle = angle/180*math.pi
      rows, cols, _ = image.shape
      if corners is None:
        corners = ((0,0), (cols, 0), (0, rows), (cols, rows))
      pivot = (rows/2, cols/2)
      
      M, min_point, max_point = self.get_rotate_matrix(corners, radian_angle, pivot)
      
      if min_point[0] != 0 or min_point[1] != 0:
          # move image if necessary
          offset_x = -min_point[0]
          offset_y = -min_point[1]
          M, min_point, max_point = self.get_rotate_matrix(corners, radian_angle, pivot, offset_x, offset_y)
      
      image = cv2.warpAffine(image,M,(int(max_point[0]), int(max_point[1])))
      new_corners = self.get_rotated_point(M, corners)
      return image, new_corners

  
def main():
  image_manipulation = Image_manipulation()
  image = cv2.imread('source-data/test.png')
  image, new_corners = image_manipulation.rotate_bound(image, 45)
  # print('rotated corners:', new_corners)
  cv2.imshow('image', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
if __name__ == '__main__':
  main()