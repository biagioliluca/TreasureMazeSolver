import sys
import numpy as np
import math
from matplotlib import pyplot as plt
import cv2

def get_digits(image_path):
  '''
    Function that extracts digits images from grid-image and return them
  '''
  
  img = cv2.imread(image_path)

  # 1. get the negative image
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  _, img_th = cv2.threshold(gray,127,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
  negative = 255 - img_th

  # 2. find the grid contour and crop the image to zoom it
  contours, hierarchy = cv2.findContours(negative, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
  x,y,w,h = cv2.boundingRect(max_contour)

  negative_cropped = negative[y:y+h, x:x+w]
  negative_cropped = cv2.resize(negative_cropped, (0,0), fx=5, fy=5)

  # 3 create a negative_cropped copy 
  negative_cropped_copy = negative_cropped.copy()
  
  # 4. create a rect kernel to dilate the image and blur it
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
  negative_cropped_copy = cv2.dilate(negative_cropped_copy, rect_kernel, iterations=7)
  negative_cropped_copy = cv2.GaussianBlur(negative_cropped_copy, (5,5), 0)

  # 5. find the contours and extract only the digits contours (no child contours)
  contours, hierarchy = cv2.findContours(negative_cropped_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  rects = [] # rects element: [x, y, width, height]
  num_digits = 0

  img_copy = negative_cropped.copy() # used to print grid-image with detected rects 

  for i in range(len(hierarchy[0])):
    if hierarchy[0][i][2] == -1:
      x,y,w,h = cv2.boundingRect(contours[i])
      
      # check if detected rect contains a digit or not. if it doesn't, takes parent countour
      detected_rect = negative_cropped[y:y+h,x:x+w]
      if (cv2.countNonZero(detected_rect) == 0) or (((negative_cropped.shape[0]*negative_cropped.shape[1]) / (w*h)) > 6000):
        x,y,w,h = cv2.boundingRect(contours[hierarchy[0][i][3]])
      
      cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0 ,0),3)
      rects.append([x,y,w,h])
      num_digits+=1

  # check if all digits get dected
  if int(math.sqrt(num_digits)) ** 2 != num_digits:
    raise Exception("Something's wrong with the detection")

  # 6. sort digits rectangles from left to right, from up to down
  sorted_grid=[]

  rects.sort(key=lambda y: y[1]) # sort by y
  k=0
  sorted_raw=[]
  for i in range(len(rects)):
    sorted_raw.append(rects[i])
    k +=1
    if k == math.sqrt(num_digits):
      sorted_raw.sort(key=lambda x: x[0]) # sort the raw by x
      sorted_grid.append(sorted_raw)

      sorted_raw=[]
      k = 0

  # print image with drawed digits rects
  plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
  plt.show()
  return negative_cropped, sorted_grid

def extract_and_preprocess(image_path):
  '''
    Extracts digits images from grid-image and transforms them on available 1-dimension vectors 
  '''
  negative, digits = get_digits(image_path)

  digit_list = []
  # resize digits images in 28x28
  for i in range(len(digits)):
    for j in range(len(digits[0])):
      x,y,w,h = digits[i][j]
      cropped_rect = negative[y:y+h, x:x+w]
      
      processed_digit = cv2.resize(cropped_rect,(20,20))
      processed_digit = cv2.copyMakeBorder(processed_digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0,0,0))
      # apply threshold to delete nuances caused by resing
      _ , processed_digit = cv2.threshold(processed_digit,127,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

      digit_list.append(processed_digit)

  # transform the 28x28 images to 1-dimension vectors and normalize them  
  digit_array = np.asarray(digit_list)
  digit_images = digit_array

  digit_array = digit_array.reshape(digit_array.shape[0], 784)
  digit_array = digit_array.astype('float32')/255
  return digit_array, digit_images