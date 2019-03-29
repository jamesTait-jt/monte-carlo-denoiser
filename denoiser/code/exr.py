import OpenEXR
import Imath
import numpy as np

def open(filepath):
  # Check if the file is an EXR file
  if not OpenEXR.isOpenExrFile(filepath):
    raise Exception("File '%s' is not an EXR file." % filepath)
  return OpenEXR.InputFile(filepath)

def getChannelAsNumpyArray(exr_file, channel):
    # Get the header of the exr file
    header = exr_file.header()
    data_window = header["dataWindow"]
    
    # Get the size of the image
    size = (data_window.max.x - data_window.min.x + 1,
            data_window.max.y - data_window.min.y + 1)

    # All the data is stored as float
    data_type = Imath.PixelType(Imath.PixelType.FLOAT)

    channel_data = exr_file.channel(channel, data_type)
    channel_data = np.fromstring(channel_data, dtype = np.float32)
    # Numpy arrays are (row, col)
    channel_data.shape = (size[1], size[0])
    return channel_data

def getBuffer(exr_file, buffer_name):
    if buffer_name in ["albedoVariance", "diffuseVariance", "normalVariance", "depth", "depthVariance"]:
        return getBuffer1D(exr_file, buffer_name)
    else:
        return getBuffer3D(exr_file, buffer_name)

def getBuffer3D(exr_file, buffer_name):
    channel_1 = getChannelAsNumpyArray(exr_file, buffer_name + ".R")
    channel_2 = getChannelAsNumpyArray(exr_file, buffer_name + ".G")
    channel_3 = getChannelAsNumpyArray(exr_file, buffer_name + ".B")
    channel = np.dstack((channel_1, channel_2, channel_3))
    return channel

def getBuffer1D(exr_file, buffer_name):
    return getChannelAsNumpyArray(exr_file, buffer_name + ".Z")
