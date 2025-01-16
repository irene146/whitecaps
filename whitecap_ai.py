import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import io, filters, morphology
from skimage.draw import polygon 

def process_whitecap_image(image_path):
    """
    Process an image to analyze whitecaps on water surface.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    
    Returns:
    --------
    Processed image data and intermediate processing results
    """
    # Load image and convert to double precision
    im = io.imread(image_path).astype(np.float64)
    
    # Just use information from red and green channel, as blue is too bright.
    im_rg = (im[:,:,0] + im[:,:,1]) / 2
    
    # Can inspect the different RGB channels here.
    # plt.figure(1); plt.imshow(im[:,:,0]); plt.title('Red'); plt.colorbar()
    # plt.figure(2); plt.imshow(im[:,:,1]); plt.title('Green'); plt.colorbar()
    # plt.figure(3); plt.imshow(im[:,:,2]); plt.title('Blue'); plt.colorbar()
    # 
    # plt.figure(4); plt.imshow(np.mean(im, axis=2)); plt.title('Grayscale'); plt.colorbar()
    # plt.figure(5); plt.imshow(im_rg); plt.title('Red-Green Average'); plt.colorbar()
    
    # Crop image to remove railing 
    # I assume ship railing? 
    im_crop = im_rg[1000:2600, :4000]
    nr, _ = im_crop.shape #nr is number of rows in cropped image 
    
    # Define the vertices of the region to blank
    BW = np.zeros(im_crop.shape, dtype=bool) #creates array of  values the same size of the image 
                                             # bianry image with all false values, 'black?)
    BW[polygon(
        [nr, 500, nr, nr], #500 is rows of polygon vertices 
        [2300, 4000, 4000, 2300], # x coordiantes for polygon vertices 
        shape=im_crop.shape #polygon coordinates  clipped to fit within the bounds of the image dimensions
    )] = True #pixels inside the polygon are set to ture (white)
            # pixels outside are set as false (black) 
#The region is blanked out to prevent it from interfering with whitecap detection

    '''
    2300   4000
   |      |
   v      v
nr +------+  <-- Top of polygon
   |      |
500+------+  <-- Bottom of polygon
   |      |
nr +------+  <-- Bottom of image

    '''

    
    # Find the pixel indices of the region to blank
    blank_indices = np.where(BW)  
    '''
    returns the indices of True values in the boolean mask BW
    It returns a tuple of two arrays:

    First array contains row indices
    Second array contains column indices
    '''
    
    # Remove background because illumination increases with distance from ship
    # to horizon. Therefore, need to "flatten" the image intensity plane
    # Set these to 2 not really understand this 
    im_crop_processed = im_crop.copy() #copy of cropped image to not modify original one 
    im_crop_processed[blank_indices] = 2 #Sets the pixel values at the specific indices to 2
                                         # This is essentially "blanking"
    




    # Calculate the background image as the minimum pixel within some moving block.
    # Here, can vary the size of each block. It needs to be large enough so it
    # is bigger than a single whitecap, but small enough to be useful.
    # There are other methods to remove the background also. 
    # See approach in Brumer paper that I emailed.
    
    # calculating a moving (sliding) minimum across the image.
    # Creates a background estimation by finding the minimum pixel value in a local neighborhood
    # 
    
    def moving_min(block):
        '''
        np.nanmin(block): Finds the minimum value in the block, ignoring NaN values
        np.ones_like(block): Creates an array of the same shape as the input, filled with ones
        Multiplies the minimum value by an array of ones, effectively creating a block where every pixel is set to the minimum value
        '''
        return np.nanmin(block) * np.ones_like(block)
    

    # Apply moving minimum to remove background
    background = ndi.generic_filter(im_crop_processed, moving_min, size=(150, 150))# adjust size of block
    '''
    Creates a background estimation by finding minimum values in 150x150 pixel blocks
    Each block is replaced with its minimum value
    '''
    
    # Reset blanked regions, bakcgroudn and imagaes 
    im_crop_processed[blank_indices] = np.nan
    background[blank_indices] = np.nan
    
    # Can try to smooth out the BG image with a Gaussian filter
    # This step does introduce some edge effects near the blanked portion of the image
    background_smoothed = filters.gaussian(background, sigma=30) #singma is amount of smoothing
    
    # Subtract background
    im_crop_subtracted = im_crop_processed - background_smoothed
    #Removes large-scale illumination variations
    
    # Display the images with and without background subtraction
    plt.figure(1)
    plt.clf()
    plt.title('Image without background removed')
    plt.imshow(im_crop)
    plt.colorbar()
    
    # This one should be somewhat more uniform.
    # The smaller the initial cropped image, the less noticeable the effect is.
    plt.figure(2)
    plt.clf()
    plt.title('Image with background removed (Note edge effects of Filter)')
    plt.imshow(im_crop_subtracted)
    plt.colorbar()
    
    # Calculate the Image Structure and PIP as outlined in Callaghan and White (2009)
    # Set up an intensity vector with the max val somewhere near the max intensity
    intensity_vector = np.arange(0, np.nanmax(im_crop_subtracted)*0.9, 0.01)
    '''
    np.nanmax(im_crop_subtracted): Finds the maximum intensity value in the image, ignoring NaN values.
    *0.9: Multiplies by 0.9 to avoid extreme outliers at the maximum intensity.
    np.arange(start, stop, step): Creates a range of intensity thresholds from 0 to 90% of the max intensity with a step size of 0.01. 
    '''
    
    # Calculate the PIP
    hist, bin_edges = np.histogram(im_crop_subtracted[~np.isnan(im_crop_subtracted)], 
                                   bins=intensity_vector)
    
    '''
    Counts how many pixel intensities fall within each range (bin) defined by the intensty vector
     Excludes NaN values to ensure valid calculations.
    np.histogram: Computes the histogram, returning:
    hist: Counts of pixels in each bin.
    bin_edges: The boundaries of the bins.
    '''
    
    # Flip to start at highest intensity threshold to lowest
    hist_reversed = hist[::-1]
    bin_edges_reversed = bin_edges[::-1]
    
    # Get cumulative sum at each successively lower intensity
    cumulative_sum = np.cumsum(hist_reversed)
    
    # Calculate PIP
    pip = np.gradient(cumulative_sum) / cumulative_sum[:-1]
     #Derives the PIP, which measures how the cumulative sum changes with respect to intensity.
     #rate of change of pixel counts
     #Normalizes by the cumulative sum, excluding the last value to match dimensions.

    
    
    # Smooth PIP
    pip_smoothed = filters.gaussian_filter1d(pip, sigma=5)
    
    # Calculate first and second derivatives
    grad_pip = np.gradient(pip_smoothed[::-1]) #reversed bc reversed histogram
    grad_pip_second = np.gradient(filters.gaussian_filter1d(grad_pip, sigma=5))#smooths derivative, then differientiates again 
    
    # Display the image structure
    # Here, the transition between whitecap to background water occurs where you
    # see the very sharp rise in PIP value as pixel intensity decreases. 
    # The transition is continuous and not a step change because of hazy 
    # foam patches.
    plt.figure(3)
    plt.clf()
    plt.plot(bin_edges_reversed[:-1], pip, '-ko')
    plt.plot(bin_edges_reversed[:-1], pip_smoothed, '-r.')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('PIP')
    
    plt.figure(4)
    plt.plot(bin_edges_reversed[:-1], grad_pip_second, '-ko')
    
    # Binary threshold for whitecaps
    whitecap_mask = im_crop_subtracted > 0.42
    
    plt.figure(5)
    plt.imshow(whitecap_mask, cmap='binary')
    
    plt.figure(6)
    plt.imshow(im_crop_subtracted)
    
    # The task now is to automate the location of the peak in the 2nd
    # derivative. This task is best done when you have become more familiar with
    # the image processing, and the setup of your camera.
    # The algorithm will not always work because the ship moves, ambient light
    # conditions change, and whitecap sizes change. Unfortunately, automation of 
    # an image processing routine is not a trivial task, and automatically
    # identifying whitecaps correctly requires a great deal of effort.
    
    # Be careful with the background removal, as it can also remove whitecap
    # signal.
    
    # Note in this example, the polariser reduced some of the intensity of the
    # whitecap in the middleground
    
    # Prepare return dictionary with processing results
    return {
        'original_image': im,
        'red_green_avg': im_rg,
        'cropped_image': im_crop,
        'background_removed': im_crop_subtracted,
        'pip': pip,
        'pip_smoothed': pip_smoothed,
        'whitecap_mask': whitecap_mask,
        'gradient_pip': grad_pip_second
    }

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = '../Images/IMG_7355.JPG'
    results = process_whitecap_image(image_path)
    plt.show()