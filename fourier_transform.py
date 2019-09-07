# import cv # use this instead of the line below
from cv2 import cv2 # this ignores vscode bug of not detecting functions form cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def plt_show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def cv2_show_img(img):
    ANY_KEY = 0
    cv2.imshow('', img)
    cv2.waitKey(ANY_KEY)

def apply_fourier_transform(img):
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

def get_spectrum(img):
    dft_shift = apply_fourier_transform(img)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    return magnitude_spectrum

def apply_mask(img, mask):
    dft_shift = apply_fourier_transform(img)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back

def get_hpf_mask(img):
    '''
        Create a high pass filter mask
    '''
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    return mask

def get_lpf_mask(img):
    '''
        Create a low pass filter mask
    '''
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1

    return mask

def get_bpf_mask(img):
    '''
        Create a band pass filter mask,
    '''
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    r_out = 80
    r_in = 10
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                            ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1

    return mask

def apply_lpf(img):
    '''
        Apply a low pass filter to the image
    '''
    mask = get_lpf_mask(img)
    return apply_mask(img, mask)

def apply_hpf(img):
    '''
        Apply a high pass filter to the image
    '''
    mask = get_hpf_mask(img)
    return apply_mask(img, mask)

def apply_bpf(img):
    '''
        Apply a ban pass filter to the image
    '''
    mask = get_hpf_mask(img)
    return apply_mask(img, mask)

def optimize_img_for_DFT(img):
    '''
        Return an image with optinal size to speed up the fourier transform
    '''
    rows, cols = img.shape
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)

    nimg = np.zeros((nrows,ncols))
    nimg[:rows,:cols] = img

    return nimg


if __name__ == '__main__':
    #img = cv2.imread('images/ciel_bw.jpg', cv2.IMREAD_GRAYSCALE)
    #img = optimize_img_for_DFT(img)

    #spectrum = get_spectrum(img)
    #hpf = apply_hpf(img)
    #lpf = apply_lpf(img)
    #bpf = apply_bpf(img)

    #img_comparison = np.concatenate((img, spectrum), axis=1)
    #plt_show_img(hpf)

    
    def grab_frame(cap):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = optimize_img_for_DFT(frame)
        spectrum = get_spectrum(frame)

        img_comparison = np.concatenate((frame, spectrum), axis=1)
        return img_comparison

    DEFAULT_CAM = 0
    cap = cv2.VideoCapture(DEFAULT_CAM)
    frame_view = plt.imshow(grab_frame(cap), cmap='gray')

    def update(i):
        frame_view.set_data(grab_frame(cap))

    ani = FuncAnimation(plt.gcf(), update, interval=200)
    plt.show()
    

'''
    cv2.imshow not working
    DEFAULT_CAM = 0
    cap = cv2.VideoCapture(DEFAULT_CAM)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        spectrum = apply_hpf(gray)
        plt_show_img(spectrum)

        # Display the resulting frame
        #cv2.imshow('frame', gray)
        #if cv2.waitKey(0):
         #   break

    cap.release()
    cv2.destroyAllWindows()
'''
