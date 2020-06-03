import cv2, os, glob
from PIL import Image
import numpy as np
import cv2
from flask import render_template, send_from_directory, request, redirect, jsonify
import base64
import numpy as np
from PIL import Image
import flask
from flask_cors import CORS
import json, math
import codecs
from io import BytesIO
from download_data import DataDownload
import lane_processing as lp 
import sys
# sys.setrecursionlimit(44500000000)
from svg.path import parse_path
from svg.path.path import Line
from svg.path.path import CubicBezier
from xml.dom import minidom
import geog
import requests
from PIL import Image
import shapely
from shapely import geometry
import geopandas, geojson
import geopandas as gp
from skimage.morphology import skeletonize, medial_axis

from skimage.util import invert
from skimage import filters
from skimage import data
from scipy.ndimage import generic_filter

import tempfile, multiproc
#http://192.168.2.181:8888/edit/Marks_Detection_Demo/untitled.txt#essing, requests, geojson
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from multiprocessing.dummy import Pool
from contextlib import closing



def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def get_hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=300):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def probabilistic_hough(edges, threshold=10, line_length=5, line_gap=3):
    lines = probabilistic_hough_line(edges, threshold=threshold, line_length=line_length, line_gap=line_gap)
    return lines

def draw_lines(img, lines, color=[255, 255, 255], thickness=1):
 
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
    return img

def draw_lines2(zeros, lines):
    for line in lines:
        lines_ = []
        for l in line:
            lines_.append((l))
    #         print(lines_)
        lines_ = np.array(lines_)
    #     print(lines_)
        cv2.polylines(zeros, [lines_], True, (0, 255, 0), 2)
    return zeros

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def convertZoom(xtile, ytile, new_zoom=18, old_zoom=20):
#     xtile, ytile, new_zoom, old_zoom = int(xtile), int(ytile), int(new_zoom), int(old_zoom)
    n = 2.0 ** old_zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    lat_rad = math.radians(lat_deg)

    n2 = 2.0 ** new_zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n2)
    #print(lon_deg, lat_deg)
    ytile = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n2)
    return (int(xtile), int(ytile))


def tile2xypixel(xpoint,ypoint,xtile, ytile,tile_size= 512):
    x = (int((xpoint - xtile) * 512))
    y = (int((ypoint - ytile) * 512))
    return x ,y

def lonlat2xy(lon_deg, lat_deg, zoom=18): 
#     zoom = int(zoom)
    lon_deg, lat_deg = float(lon_deg), float(lat_deg)
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n)
    ytile = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)



def xy2lonlat(xtile, ytile, zoom=18):
    zoom = (zoom)
    xtile, ytile = (xtile), (ytile)
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return str(lat_deg), str(lon_deg)



def find_road_lines(image):
    kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    ret3,th3 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

def merge_tiles(prediction_list, MINX, MINY, MAXX, MAXY):
    full_image = np.array(prediction_list)
    rows = (MAXX+1) - MINX
    cols = (MAXY+1) - MINY
    image_size = prediction_list[0].shape[0]
    image_channels = len(prediction_list[0].shape)
    if image_channels == 3:
        full_image = full_image.reshape(rows*cols*image_size, image_size,3)
    else:
        full_image = full_image.reshape(rows*cols*image_size, image_size)
    temp = None

    for i in range(rows):
        start = (i) * cols * image_size
        end = (i+1) * cols * image_size
        if temp is None:            
            temp = full_image[start: end, :]
        else:            
            temp = np.concatenate((temp, full_image[start: end, :]), axis=1)

    full_image = None
    return temp 



def polygons_to_mask_(polygons, im_size, bg=False, polylines=False,color = 255,linewidth=2):
    """convert sheply polygon to mask or draw on image"""
    if bg is None:
        img_mask = np.zeros(im_size, np.uint8)
    else:
        img_mask = bg.copy()
    if not polygons:
        return img_mask
    def int_coords(x): return np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
        
    if polylines:
        cv2.polylines(img_mask,exteriors,True,color)
    else:
        cv2.fillPoly(img_mask, exteriors, color)
        cv2.fillPoly(img_mask, interiors, 0)
    return img_mask



def polygonize_contours_opencv(image,simValue=.1):
    """
    Convert openCV contours to shapely's polygon
    Args:
        array: numpy contours

    Returns:
        list of shapely.geometry.Polygon
    """
    _, contours = cv2.findContours(image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        n,  k = cnt.shape
        if n > 2:
#             if  cv2.contourArea(cnt) > area :
            cnt = cnt.reshape(n, k)
            # hull_polygon = shapely.geometry.Polygon(cnt)
            hull_polygon = geometry.Polygon(cnt).simplify(simValue, preserve_topology=True)

            polygons.append(hull_polygon)

    return polygons



def convert_crs_json(json_data,fromlon,fromlat,ZOOM_LEVEL):
    fromx,fromy  = lonlat2xy(fromlon,fromlat,ZOOM_LEVEL)
    outise_polys = []
    inside_polys = []
    for feature in json_data['features']:
        point_list = [lonlat2xy(x,y,ZOOM_LEVEL) for (x,y)  in feature['geometry']['coordinates'][0]]
        point_list = [tile2xypixel(x,y,int(fromx),int(fromy),512) for (x,y)  in point_list]
        outise_polys.append(geometry.Polygon(point_list))


        for i in range(1,len(feature['geometry']['coordinates'])):
            point_list = [lonlat2xy(x,y,ZOOM_LEVEL) for (x,y)  in feature['geometry']['coordinates'][i]]
            point_list = [tile2xypixel(x,y,int(fromx),int(fromy),512) for (x,y)  in point_list]
            inside_polys.append(geometry.Polygon(point_list))
    return outise_polys, inside_polys



def get_sat_image(fromx,fromy ,tox,toy,zoom, data_source):
    data_folder = "./data/"
    dataDownload = DataDownload(fromx,fromy ,tox,toy , "CUSTOM", "LAYERUID", 
                                zoom,data_folder, DATA_SOURCE_URL=data_source[:-1])
    dataDownload.get_data()
    sat_images_list = []
    for x in range(fromx, tox+1):
        for y in range(fromy, toy+1):
            img = cv2.cvtColor(cv2.imread("%s%s_%s_%s.png"%(data_folder,zoom,x,y)), cv2.COLOR_BGR2RGB)
            sat_images_list.append(img)
    return sat_images_list

def xypixel2tile(x, y, xtile, ytile, size=512):
    xpoint = (x/size + xtile)
    ypoint = (y/size + ytile)
    
    return xpoint, ypoint


def get_hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=300):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines
def draw_lines(img, lines, color=[255, 255, 255], thickness=1):
    
    
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    if (type(lines) != np.ndarray):
        return img
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img

def grayscale(img):
    
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def hough_lines(img, rho=1, min_line_len=13,threshold=70, max_line_gap=300):
#     img = grayscale(img)
#     print(img.shape)
    edges = cv2.Canny(img, 80, 200)
    hough = get_hough_lines(edges, rho=rho, min_line_len=min_line_len, threshold=threshold, max_line_gap=max_line_gap)
    canvas = np.zeros_like(img)
    return draw_lines(canvas, hough)