import numpy as np
from math import pi
import matplotlib.pyplot as plt
from numpy import _ArrayComplex_co, newaxis
import time
def disp(x):
    with np.printoptions(precision=2,suppress=Time):
        print(x)


def gen_dft_mtx(N,disp_flag=1):
    row_index=np.array([0,1,2,3]) 
    col_index=row_index.T


    if(disp_flag):
        print("\n row index=",row_index)
        print("\n row index shape:",row_index.shape)

        print("col index=\n",col_index)
        print("col index shape:",col_index.shape)
        

    index_matrix=col_index@row_index 
    dft_mtx=np.exp(-1j*2*pi*index_matrix/N)
    if(disp_flag):
        print("\n\n index matrix:\n",index_matrix)
        print("\n DFT matrix:\n")

        disp(dft_mtx)
    return(dft_mtx)
   N=4
   dft_mtx=gen_dft_mtx(N,1)





















































hello alex
















