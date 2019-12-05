#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:21:26 2018

@author: rossspencer
"""

from skimage.measure import regionprops
import numpy as np

def bounding_box(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	if np.where(rows)[0].shape[0] == 0 or np.where(cols)[0].shape[0] == 0:
		return 0,0,0,0
	ymax, ymin = np.where(rows)[0][[0, -1]]
	xmin, xmax = np.where(cols)[0][[0, -1]]
	return ymax, ymin, xmin, xmax

def centroid(img):
	labeled_foreground = (img > 0).astype(int) ##threshold value := 0
	properties = regionprops(labeled_foreground, img)
	#center_of_mass = properties[0].centroid
	center_of_mass = properties[0].weighted_centroid
	return center_of_mass

def box_center(img):
	ymax, ymin, xmin, xmax = bounding_box(img)
	if (bounding_box(img) == (0,0,0,0)):
		return 0,0,0,0
	horiz = xmin + (xmax-xmin)/2
	vert = ymax + (ymin-ymax)/2
	width = xmax - xmin + 1
	height = ymin - ymax + 1
	#   feature 1, feature 2, feature 3, feature 4
	return horiz, vert, width, height

def num_on(img):
	# feature 5
	return np.sum(img)


def avg(img):
	on = np.where(img == 1)
	horiz, vert, width, height = box_center(img)
	on0 = [y - vert for y in on[0]]
	on1 = [x - horiz for x in on[1]]
	#feature 6
	avg_x = (np.sum(on1)/(width*len(on1)))
	#feature 7
	avg_y = (np.sum(on0)/(height*len(on0)))
	#feature 8
	ssq_x = np.sum(np.array(on1)**2)/len(on1)
	#feature 9
	ssq_y = np.sum(np.array(on0)**2)/len(on0)
	#feature 10
	s_xy = -(np.array(on1) @ np.array(on0))/len(on1)
	#feature 11
	corr_horizvar_w_vertipos = sum(ssq_x*np.array(on0))/len(on0)
	#feature 12
	corr_vertivar_w_horizpos = sum(ssq_y*np.array(on1))/len(on1)

	# feature 6, feature 7, feature 8, feature 9, feature 10, feature 11, feature 12
	return avg_x, avg_y, ssq_x, ssq_y, s_xy, corr_horizvar_w_vertipos, corr_vertivar_w_horizpos


## len(img[0,:]) == 143
## len(img[:,0]) == 66
	## last 4 features to extract, need to go thru the horizontal edges and vertical edges
def horiz_edges(img):
	ymax, ymin, xmin, xmax = bounding_box(img) #get the bounding box
	xedges = 0
	vertilocation_sum = 0
	xscans = 0
	for j in range(ymax, ymin):
		for i in range(xmin, xmax):
			if img[j][i] == 0 and img[j][i+1] == 1: # to the right?
				xedges += 1
				vertilocation_sum += (j+1)
		xscans += 1
	yedges = 0
	horilocation = 0
	yscans = 0
	for i in range(xmin, xmax):
		for j in range(ymax, ymin):
			if img[j][i] == 0 and img[j-1][i] == 1: # top?
				yedges += 1
				horilocation += (i+1)
		yscans += 1

	if xscans == 0:
		xscans += 1
	if yscans == 0:
		yscans += 1
	meanxedges = xedges/xscans
	meanyedges = yedges/yscans
	return meanxedges, vertilocation_sum, meanyedges, horilocation
