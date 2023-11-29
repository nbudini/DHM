#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:45:03 2023

@author: nbudini
"""
# Phase reconstruction from a digital plane-wave off-axis hologram
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
from skimage.restoration import unwrap_phase

# load the hologram as a grayscale image
filename = 'gota0040.bmp'
holo = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
nfil, ncol = holo.shape # get image size
 
# convolutional high-pass filter to remove DC term
flevel = 25 # lower value -> higher cutoff frequency
kernel = np.ones(flevel)/flevel**2
holo_f = np.double(holo) - np.mean(np.double(holo))

# show original and filtered hologram
fig, ax = plt.subplots(1,2,tight_layout=True)
ax[0].imshow(holo,cmap=mpl.colormaps['turbo'])
ax[0].axis("off")
ax[1].imshow(holo_f,cmap=mpl.colormaps['turbo'])
ax[1].axis("off")

# frequency spectrum (angular spectrum)
holo_ft = np.fft.fftshift(holo_f)
holo_ft = np.fft.fft2(holo_ft)
holo_ft = np.fft.fftshift(holo_ft)

# show angular spectrum
plt.imshow(np.log(np.abs(holo_ft)))

# pick up +1 or -1 order row and column
filMax, colMax = np.unravel_index(np.argmax(np.abs(holo_ft), axis=None), holo_ft.shape)

# radius of mask around +1 or -1 order
r = np.abs(colMax-ncol/2)*2/4 
#r = 148; # or set it manually

# generate mask
mask = np.zeros(holo.shape);
for col in np.arange(0,ncol):
    for fil in np.arange(0,nfil):
        if np.sqrt((col-colMax)**2+(fil-filMax)**2)<r:
            mask[fil,col] = 1.0

# apply mask
holo_ft_m = holo_ft*mask

# center carrier frequency
rowsToShift = round(nfil/2 - filMax)
colsToShift = round(ncol/2 - colMax)
holo_ft_ms = np.roll(holo_ft_m,rowsToShift,axis=0) # rows
holo_ft_ms = np.roll(holo_ft_ms,colsToShift,axis=1) # columns
 
# angular spectrum propagation parameters
pixh = 1.55e-6 # pixel horizontal size
pixv = 1.55e-6 # pixel vertical size
lda = 632.8e-9 # wavelength
k = 2*np.pi/lda # wavenumber
ximg = np.linspace(-ncol/2,ncol/2,ncol)*pixh # x scale
yimg = np.linspace(-nfil/2,nfil/2,nfil)*pixv # y scale
[X,Y] = np.meshgrid(ximg,yimg) # coordinates grid at propagation plane
fm = 1/(nfil*pixv**2) # y frequency step
fn = 1/(ncol*pixh**2) # x frequency step
corrfase = k*np.sqrt(1-(lda*fm*Y)**2-(lda*fn*X)**2) # phase variation due to propagation

# propagation distance
dopt = 16.3e-2 # it might be + or -

# propagated field
prop = np.exp(1j*corrfase*dopt)
img_rec = np.fft.ifftshift(prop*holo_ft_ms)
img_rec = np.fft.ifft2(img_rec)
img_rec = np.fft.ifftshift(img_rec)

# amplitude (modulus) and wrapped phase
mod_rec = np.abs(img_rec); # modulus
fase_rec = np.angle(img_rec) # wrapped phase

# amplitud and phase plots
fig, ax = plt.subplots(1,2,tight_layout=True)
ax[0].imshow(mod_rec,cmap=mpl.colormaps['turbo'])
ax[0].axis("off")
ax[1].imshow(fase_rec,cmap=mpl.colormaps['turbo'])
ax[1].axis("off")

# phase unwrapping
fase_unw = -unwrap_phase(fase_rec) # minus sign just makes heights positive (omit if not needed)

# phase 3D plot
fig = plt.figure()
ax = plt.axes(projection ='3d')
 
# Creating plot
ax.plot_surface(X,Y,fase_unw) 
 
# show plot
plt.show()