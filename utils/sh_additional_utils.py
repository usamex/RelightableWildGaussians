'''
MIT License

Copyright (c) 2018 Andrew Chalmers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
Utility functions
'''

import os
import numpy as np
import math
import argparse
import imageio.v3 as im
import cv2 # resize images with float support
from scipy import ndimage # gaussian blur
import skimage.measure # max_pooling with block_reduce
import time
import utils.spherical_harmonics as  sh


def blur_ibl(ibl, amount=5):
	x = ibl.copy()
	x[:,:,0] = ndimage.gaussian_filter(ibl[:,:,0], sigma=amount)
	x[:,:,1] = ndimage.gaussian_filter(ibl[:,:,1], sigma=amount)
	x[:,:,2] = ndimage.gaussian_filter(ibl[:,:,2], sigma=amount)
	return x

def resize_image(img, width, height, interpolation=cv2.INTER_CUBIC):
	if img.shape[1]<width: # up res
		if interpolation=='max_pooling':
			return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
		else:
			return cv2.resize(img, (width, height), interpolation=interpolation)
	if interpolation=='max_pooling': # down res, max pooling
		try:
			scale_factor = int(float(img.shape[1])/width)
			factored_width = width*scale_factor
			img = cv2.resize(img, (factored_width, int(factored_width/2)), interpolation=cv2.INTER_CUBIC)
			block_size = scale_factor
			r = skimage.measure.block_reduce(img[:,:,0], (block_size,block_size), np.max)
			g = skimage.measure.block_reduce(img[:,:,1], (block_size,block_size), np.max)
			b = skimage.measure.block_reduce(img[:,:,2], (block_size,block_size), np.max)
			img = np.dstack((np.dstack((r,g)),b)).astype(np.float32)
			return img
		except:
			print("Failed to do max_pooling, using default")
			return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
	else: # down res, using interpolation
		return cv2.resize(img, (width, height), interpolation=interpolation)

def get_solid_angle(y, width, is3D=False):
	"""
	y = y pixel position (cast as a float)
	Solid angles in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scale_factors.pdf
	"""
	height = int(width/2)
	pi2_over_width = (np.pi*2)/width
	pi_over_height = np.pi/height
	theta = (1.0 - ((y + 0.5) / height)) * np.pi
	return pi2_over_width * (np.cos(theta - (pi_over_height / 2.0)) - np.cos(theta + (pi_over_height / 2.0)))

def get_solid_angle_map(width):
	height = int(width/2)
	return np.repeat(get_solid_angle(np.arange(0,height), width)[:, np.newaxis], width, axis=1)

def get_coefficients_matrix(xres,l_max=2):
	yres = int(xres/2)
	# setup fast vectorisation
	x = np.arange(0,xres)
	y = np.arange(0,yres).reshape(yres,1)

	# Setup polar coordinates
	lat_lon = xy_to_ll(x,y,xres,yres)

	# Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
	Ylm = sh.sh_evaluate(lat_lon[0], lat_lon[1], l_max)
	return Ylm

def get_coefficients_from_image(ibl, l_max=2, resize_width=None, filder_amount=None):
	# Resize if necessary (I recommend it for large images)
	if resize_width is not None:
		#ibl = cv2.resize(ibl, dsize=(resize_width,int(resize_width/2)), interpolation=cv2.INTER_CUBIC)
		ibl = resize_image(ibl, resize_width, int(resize_width/2), cv2.INTER_CUBIC)
	elif ibl.shape[1] > 1000:
		#print("Input resolution is large, reducing for efficiency")
		#ibl = cv2.resize(ibl, dsize=(1000,500), interpolation=cv2.INTER_CUBIC)
		ibl = resize_image(ibl, 1000, 500, cv2.INTER_CUBIC)
	xres = ibl.shape[1]
	yres = ibl.shape[0]

	# Pre-filtering, windowing
	if filder_amount is not None:
		ibl = blur_ibl(ibl, amount=filder_amount)

	# Compute sh coefficients
	sh_basis_matrix = get_coefficients_matrix(xres,l_max)

	# Sampling weights
	solid_angles = get_solid_angle_map(xres)

	# Project IBL into SH basis
	n_coeffs = sh.sh_terms(l_max)
	ibl_coeffs = np.zeros((n_coeffs,3))
	for i in range(0,sh.sh_terms(l_max)):
		ibl_coeffs[i,0] = np.sum(ibl[:,:,0]*sh_basis_matrix[:,:,i]*solid_angles)
		ibl_coeffs[i,1] = np.sum(ibl[:,:,1]*sh_basis_matrix[:,:,i]*solid_angles)
		ibl_coeffs[i,2] = np.sum(ibl[:,:,2]*sh_basis_matrix[:,:,i]*solid_angles)

	return ibl_coeffs

# Spherical harmonics reconstruction
def get_diffuse_coefficients(l_max):
	# From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
	diffuse_coeffs = [np.pi, (2*np.pi)/3]
	for l in range(2,l_max+1):
		if l%2==0:
			a = (-1.)**((l/2.)-1.)
			b = (l+2.)*(l-1.)
			#c = float(np.math.factorial(l)) / (2**l * np.math.factorial(l/2)**2)
			c = math.factorial(int(l)) / (2**l * math.factorial(int(l//2))**2)
			#s = ((2*l+1)/(4*np.pi))**0.5
			diffuse_coeffs.append(2*np.pi*(a/b)*c)
		else:
			diffuse_coeffs.append(0)
	return np.asarray(diffuse_coeffs) / np.pi

def sh_render(ibl_coeffs, width=600):
	l_max = sh_l_max_from_terms(ibl_coeffs.shape[0])
	diffuse_coeffs =get_diffuse_coefficients(l_max)
	sh_basis_matrix = get_coefficients_matrix(width,l_max)
	rendered_image = np.zeros((int(width/2),width,3))
	for idx in range(0,ibl_coeffs.shape[0]):
		l = l_from_idx(idx)
		coeff_rgb = diffuse_coeffs[l] * ibl_coeffs[idx,:]
		rendered_image[:,:,0] += sh_basis_matrix[:,:,idx] * coeff_rgb[0]
		rendered_image[:,:,1] += sh_basis_matrix[:,:,idx] * coeff_rgb[1]
		rendered_image[:,:,2] += sh_basis_matrix[:,:,idx] * coeff_rgb[2]
	return rendered_image

def sh_l_max_from_terms(terms):
	return int(np.sqrt(terms)-1)

def l_from_idx(idx):
	return int(np.sqrt(idx))

def xy_to_ll(x,y,width,height):
	def yLocToLat(yLoc, height):
		return (yLoc / (float(height)/np.pi))
	def xLocToLon(xLoc, width):
		return (xLoc / (float(width)/(np.pi * 2)))
	return np.asarray([yLocToLat(y, height), xLocToLon(x, width)], dtype=object)
