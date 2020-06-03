import cv2,os,glob, shutil, json, PIL, random
import numpy as np
from shapely.geometry import *
from shapely import wkt
from PIL import Image
from cv2 import imread, imwrite, flip
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely import geometry
from skimage.morphology import skeletonize, skeletonize_3d
from skimage import data
import matplotlib.pyplot as plt
# from skimage.util import invert


def makedirs(_dirs):
    for _dir in _dirs:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
def getName(path):
    return os.path.basename(path)

def getFiles(path):
    path =  path if path[-1] == '/' else (path + '/')
    return glob.glob(path + '*')

def skeleton(image):
    if len(image.shape) == 3:
        image = image[:, :, 0]
    image[image == 255] = 1
    skel = skeletonize(image)
    skel = skel.astype(int)
#     skel[skel == 0] = 255
    skel[skel == 1] = 255
    skel = skel.astype(np.uint8)
    skel = np.stack([skel]*3, axis=-1)
    return skel