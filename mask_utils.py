import numpy as np

def upper_body_mask(image):
  mask1 = image ==1
  mask2 =  image %2==0
  tensor = np.logical_or(mask1, mask2)
  return tensor.astype(np.int8)

def lower_body_mask(image):
  mask1 = image !=1
  mask2 =  image %2!=0
  
  tensor = np.logical_and(mask1, mask2)
  return tensor.astype(np.int8)

def get_body_mask(image):
  mask = image >0
  return mask.astype(np.int8)