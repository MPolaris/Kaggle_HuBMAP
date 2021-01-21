import os
import tifffile
import numpy as np

def get_data(path):
    image = tifffile.imread(path)
    if image.ndim == 5:
        image = image[0,0,:,:,:]
        image = np.transpose(image, (1,2,0))
    return image

def make_grid(shape, window=512, min_overlap=128):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)
    
    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]    
    return slices.reshape(nx*ny,4)

def mask2rle(mask):
    ''' takes a 2d boolean numpy array and turns it into a space-delimited RLE string '''
    mask = mask.T.reshape(-1) # make 1D, column-first
    mask = np.pad(mask, 1, mode="constant") # make sure that the 1d mask starts and ends with a 0
    starts = np.nonzero((~mask[:-1]) & mask[1:])[0] # start points
    ends = np.nonzero(mask[:-1] & (~mask[1:]))[0] # end points
    rle = np.empty(2 * starts.size, dtype=int) # interlacing...
    rle[0::2] = starts + 1# ...starts...
    rle[1::2] = ends - starts # ...and lengths
    rle = ' '.join([ str(elem) for elem in rle ]) # turn into space-separated string
    return rle

def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
