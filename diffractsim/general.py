import numpy as np
import tensorflow as tf
import diffractsim
from diffractsim import MonochromaticField, mm, nm, cm

def noramlization(data):
  minVals = data.min(0)
  maxVals = data.max(0)
  ranges = maxVals - minVals
  normData = np.zeros(np.shape(data))
  m = data.shape[0]
  normData = data - np.tile(minVals, (m, 1))
  normData = normData/np.tile(ranges, (m, 1))
  return normData

def noramlizationTF(data):
  minVals = tf.reduce_min(data)
  maxVals = tf.reduce_max(data)
  normData = (data - minVals) / (maxVals - minVals)
  return normData

def holoToRgb(holo):
    diffractsim.set_backend("CPU")
    holo = holo[0, ::, ::, 0]
    XM = 128 * 1
    YN = 64 * 1
    X = XM * 0.0108
    Y = YN * 0.0108
    F = MonochromaticField(wavelength=632.8 * nm, extent_x=X * mm, extent_y=Y * mm, Nx=XM, Ny=YN)
    F.add_aperture_from_array(holo, image_size=(X * mm, Y * mm))
    recon = F.compute_colors_at(-50 * mm)
    recon = RgbToGray(recon)
    rgbGrayNor = noramlization(recon)
    rgbGrayNor4D = rgbGrayNor.reshape(1, YN, XM, 1)
    return rgbGrayNor4D

def RgbToGray(rgb):
  imgR = rgb[:, :, 0]
  imgG = rgb[:, :, 1]
  imgB = rgb[:, :, 2]
  rgbGray = 0.2990 * imgR + 0.5870 * imgG + 0.1140 * imgB
  return rgbGray