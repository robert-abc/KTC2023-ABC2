import numpy as np
from PIL import Image

def save_img(img_np, fname):
  img_np = normalize(img_np)
  ar = np.clip(img_np*255,0,255).astype(np.uint8)

  img = Image.fromarray(ar)
  img.save(fname)

def normalize(x):
    return (x-x.min())/(x.max()-x.min())