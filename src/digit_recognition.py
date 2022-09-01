import sys
import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
import keras

np.set_printoptions(threshold=sys.maxsize)

def get_digits(image_path):
  img = cv2.imread(image_path)

  # 1. get the negative image
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  ret,img_th = cv2.threshold(gray,127,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
  negative = 255 - img_th

  # 2. find the grid contour and crop the image to zoom it
  contours, hierarchy = cv2.findContours(negative, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
  x,y,w,h = cv2.boundingRect(max_contour)
  cv2.rectangle(negative, (x,y), (x+w,y+h), (0,255,0), 2)
  negative_cropped = negative[y:y+h, x:x+w]
  negative_cropped = cv2.resize(negative_cropped, (0,0), fx=5, fy=5)

  # 3. process the negative cropped version image to extract grid digits 
  negative_cropped_copy = negative_cropped.copy()
  
  # 3.1 create a rect kernel and erode the image
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
  negative_cropped_copy = cv2.dilate(negative_cropped_copy, rect_kernel, iterations=7)
  negative_cropped_copy = cv2.GaussianBlur(negative_cropped_copy, (5,5), 0)

  # 3.2 find the contours and extract only the digits contours (no child contours)
  contours, hierarchy = cv2.findContours(negative_cropped_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  rects=[]
  num_digits=0

  # debug
  img_copy = negative_cropped.copy()

  #print('HEIGHT OF SHAPE: {} '.format(img_copy.shape[0]))
  #print('WIDTH OF SHAPE: {} '.format(img_copy.shape[1]))

  # img.shape[0] return the height, img.shape[1] return the width
  # hierarchy[0][i]: [next, previous, first_child, parent]
  for i in range(len(hierarchy[0])):
    if hierarchy[0][i][2] == -1:
      x,y,w,h = cv2.boundingRect(contours[i])
      #print('width: {}, height: {}'.format(w,h))
      #print((img_copy.shape[0]*img_copy.shape[1]) / (w*h))

      black_condition_cropped = img_copy[y:y+h,x:x+w]
      if (cv2.countNonZero(black_condition_cropped) == 0) or (((img_copy.shape[0]*img_copy.shape[1]) / (w*h)) > 6000):
        x,y,w,h = cv2.boundingRect(contours[hierarchy[0][i][3]])

  
        

      cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0 ,0),3)
      rects.append([x,y,w,h])
      num_digits+=1

  if int(math.sqrt(num_digits)) ** 2 != num_digits:
    raise Exception("Something's wrong with the detection")

  # 4 sort digits rectangles from left to right, from up to down
  sorted_rects=[]

  # sort by y
  rects.sort(key=lambda y: y[1])
  k=0
  sorted_rects_raw=[]
  for i in range(len(rects)):
    sorted_rects_raw.append(rects[i])
    k +=1
    # TODO: need to make resilient to errors in disclosure of figures
    if k == math.sqrt(num_digits):
      # sort the raw by x
      sorted_rects_raw.sort(key=lambda x: x[0])
      sorted_rects.append(sorted_rects_raw)
      sorted_rects_raw=[]
      k = 0


  # debug instruction
  '''
  for i in range(len(sorted_rects)):
    for j in range(len(sorted_rects[0])):
      x,y,w,h = sorted_rects[i][j]
      cv2.rectangle(negative_cropped,(x,y),(x+w,y+h),(255,0 ,0),3)
      plt.imshow(negative_cropped[y:y+h, x:x+w])
      plt.show()
  '''
  plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
  plt.show()
  return negative_cropped, sorted_rects

def extract_and_preprocess(image_path):
  negative, digits = get_digits(image_path)

  digit_list = []
  # resizing of the digits images in 28x28
  for i in range(len(digits)):
    for j in range(len(digits[0])):
      x,y,w,h = digits[i][j]
      cropped_rect = negative[y:y+h, x:x+w]
      #plt.imshow(cv2.cvtColor(cropped_rect, cv2.COLOR_RGB2BGR))
      #plt.show()

      # !!! try to erode !!!
      #rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
      #cropped_rect = cv2.erode(cropped_rect, rect_kernel, iterations=2)
      
      # from 20*20 to 28*28 with padding
      processed_digit = cv2.resize(cropped_rect,(20,20))
      processed_digit = cv2.copyMakeBorder(processed_digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0,0,0))

      ret, processed_digit = cv2.threshold(processed_digit,127,255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

      

      #plt.imshow(cv2.cvtColor(processed_digit, cv2.COLOR_RGB2BGR))
      #plt.show()
      digit_list.append(processed_digit)

  # transform the 28x28 images to 1-dimension vectors and normalize them  
  digit_array = np.asarray(digit_list)
  digit_images = digit_array
  digit_array = digit_array.reshape(digit_array.shape[0], 784)
  digit_array = digit_array.astype("float32")/255
  return digit_array, digit_images