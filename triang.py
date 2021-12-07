import numpy as np
import matplotlib.pyplot as plt
import time
from math import pi
from numpy import newaxis

def disp(x):
    with np.printoptions(precision=2, suppress=True):
        print(x)
        
def gen_dft_mtx(N, disp_flag=1):
    row_index = np.array([np.arange(N)])
    col_index = row_index.T
    
    if(disp_flag):
        print('\nRow index :', row_index)
        print('Row index shape :', row_index.shape)
        print('\nCol index :\n', col_index)
        print('Col index shape :', col_index.shape)
    
    index_matrix = col_index @ row_index
    dft_mtx = np.exp(-1j * 2 * pi * index_matrix / N)
    if(disp_flag):
        print('\nIndex matrix :\n', index_matrix)
        print('\nDFT matrix :\n')
        disp(dft_mtx)
    return dft_mtx

N = 4
dft_mtx = gen_dft_mtx(N, 1)
plt.matshow(np.real(dft_mtx))
plt.title('Real part of DFT matrix')
plt.matshow(np.imag(dft_mtx))
plt.title('Imag part of DFT')
plt.show()

# Compute DFT matrix using DFT matrix
col_data = np.random.rand(N, 1)
fft_frm_dftmtx = dft_mtx @ col_data
print('\nFFT computed using DFT matrix')
disp(fft_frm_dftmtx)

# Compute DFT matrix via FFT routine
fft_frm_fft_fn = np.fft.fft(col_data.flatten())
print('\nFFT computed using FFT routine')
disp(fft_frm_fft_fn)

plt.figure(2)
plt.stem(np.abs(fft_frm_dftmtx.flatten() - fft_frm_fft_fn))
plt.ylim([-0.5, 0.5])
plt.title('Error between FFTs computed via DFT matrix and FFt routine ')
plt.show()

gama_values = np.arange(2, 13+1, 1)
dft_mtx_time = np.zeros(len(gama_values))
fft_routine_time = np.zeros(len(gama_values))
idx = 0
for gamma in gama_values:
    N= 2 ** gamma
    print('NFFT = ', N)
    cut_data = np.random.rand(N, 1)
    #generate DFT Matrix
    dft_mtx = gen_dft_mtx(N, 0)
 
    # Compute FFT via DFT matrix
    start_time = time.perf_counter()
    fft_frm_dftmtx = dft_mtx @ cut_data
    end_time = time.perf_counter()
    dft_mtx_time[idx] = end_time - start_time

    # Compute FFT via FFT routine
    start_time = time.perf_counter()
    fft_frm_fft_fn = np.fft.fft(cut_data.flatten())
    end_time = time.perf_counter()
    fft_routine_time[idx] = end_time - start_time
    idx = idx + 1

plt.plot(gama_values, dft_mtx_time, 'r', gama_values, fft_routine_time, 'k')

plt.legend(['DFT Matrix','FFT'])
plt.title('Time for DFT via FFT method & DFT matrix method')
plt.xlabel('gamma')
plt.ylabel('time elapsed in Seconds')

plt.show()