import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def foo():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft2, ifft2
    from mpl_toolkits.mplot3d import Axes3D

    """CREATING REAL AND MOMENTUM SPACES GRIDS"""
    N_x, N_y = 80, 80
    range_x, range_y = np.arange(N_x), np.arange(N_y)
    dx, dy = 0.005, 0.005
    # real space grid vectors
    xv, yv = dx * (range_x - 0.5 * N_x), dy * (range_y - 0.5 * N_y)
    dk_x, dk_y = np.pi / np.max(xv), np.pi / np.max(yv)
    # momentum space grid vectors, shifted to center for zero frequency
    k_xv, k_yv = dk_x * np.append(range_x[:N_x//2], -range_x[N_x//2:0:-1]), \
                dk_y * np.append(range_y[:N_y//2], -range_y[N_y//2:0:-1])

    # create real and momentum spaces grids
    x, y = np.meshgrid(xv, yv, sparse=False, indexing='ij')
    kx, ky = np.meshgrid(k_xv, k_yv, sparse=False, indexing='ij')
 
    """FUNCTION"""
    sigma=0.25
    f = 1/(2*np.pi*sigma) * np.exp(-0.5 * (x ** 2 + y ** 2)/sigma)
    F = fft2(f)
    """PLOTTING"""
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(x, y, np.abs(f), cmap='viridis')
    # for other plots I changed to
    fig2 = plt.figure()
    ax2 =Axes3D(fig2)
    surf = ax2.plot_surface(kx, ky, np.abs(F), cmap='viridis')
    plt.show()

def gauss_function(x, y, std_sqr):
    return math.exp(-(x ** 2 + y ** 2) / (2 * std_sqr))

def gauss_filter_matrix(variance, window_size):
    w_range = np.linspace(-(window_size - 1) / 2, (window_size - 1) / 2, window_size)
    XX, YY = np.meshgrid(w_range, w_range)

    exponent_part = np.vectorize(gauss_function)(XX, YY, variance)
    constant_part = 1 / (2 * math.pi * variance) ** 0.5

    return exponent_part * constant_part

if __name__ == '__main__':
    N_x, N_y = 80, 80
    std_sqr = 1
    window_size = 9

    f = gauss_filter_matrix(std_sqr, window_size)
    F = np.fft.fft2(f, (N_x, N_y))
    F = abs( np.fft.fftshift(F) )

    fig = plt.figure()
    ax = Axes3D(fig)
    range_x, range_y = np.arange(N_x), np.arange(N_y)
    '''
    dx, dy = 0.005, 0.005
    xv, yv = dx * (range_x - 0.5 * N_x), dy * (range_y - 0.5 * N_y)
    dk_x, dk_y = np.pi / np.max(xv), np.pi / np.max(yv)
    k_xv, k_yv = dk_x * np.append(range_x[:N_x//2], -range_x[N_x//2:0:-1]), \
                dk_y * np.append(range_y[:N_y//2], -range_y[N_y//2:0:-1])
    x, y = np.meshgrid(xv, yv, sparse=False, indexing='ij')
    '''
    x, y = np.meshgrid(range_x, range_y)
    surf = ax.plot_surface(x, y, F, cmap='viridis')
    plt.show()