import numpy as np
import matplotlib.pyplot as plt
from math import pi

def rect(t,PW):
    return(1*np.array(np.abs(t)<PW/2))

def impulse(t):
    return(1*np.array(np.abs(t)<0.01))

def ramp(t):
    return(t*np.array(t>0))

def bipolar(t,T,D):
    frac,int_part=np.modf(t%T)
    return(2*(frac<T*D)-1)

def triang(t,T):
    frac,int_part=np.modf(t%T)
    temp=T-frac
    frac[frac>T/2]=temp[frac>T/2]
    return(frac)

#rect pulse
plt.subplot(2,2,1)
t= np.linspace(-2,2,1000)
plt.plot(t,rect(t,0.5))
plt.title('Unit Rectangular Pulse of Pulse width{} sec'.format(0.5))
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')

#impulse signal
plt.subplot(2,2,2)
t=np.linspace(-2,2,1000)
plt.plot(t,impulse(t))
plt.title('unit impulse')
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')

#Ramp Signal
plt.subplot(2,2,3)
t=np.linspace(-2,2,1000)
plt.plot(t, ramp(t))
plt.title('Unit Ramp')
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')

plt.grid()
plt.tight_layout()
plt.show()


#Bipolar Signal
plt.figure(2)
plt.subplot(2,1,1)
t=np.linspace(0,4,1000)
plt.plot(t, bipolar(t,.5, .5))
plt.title('Bipolar Function')
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')


#Triangular Pulse
plt.subplot(2,1,2)
t=np.linspace(0,4,1000)
triang_pulse = triang(t,.5)
normalised_triang_pulse = triang_pulse/ np.max(triang_pulse)
plt.plot(t, normalised_triang_pulse)
plt.title('Triangular  Pulse')
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')

#testing without normalisation
#plt.subplot(2,2,3)
#t=np.linspace(0,4,1000)
#triang_pulse = triang(t,.5)
#plt.plot(t, triang(t,.5))
#plt.title('without normalisation Triangular  Pulse')
#plt.xlabel('Time in sec')
#plt.ylabel('Amplitude')


plt.grid()
plt.tight_layout()
plt.show()


#####################################################

plt.figure(3)

fc = 100
Fs = 10000
time_dur = 0.1
Ts = 1/Fs
N =  time_dur * Fs
n = np.arange(0,N,1)
x= np.cos(2*pi*fc/Fs*n)


t = n*Ts
plt.plot(t,x)
plt.title('Cosine Wave')
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')

plt.grid()
plt.show()


#########################################


plt.figure(4)
plt.stem(t[:150],x[:150])
plt.title('Discrete Cosine Wave')
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')
plt.grid()
plt.show()