import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import pandas as pd
import scipy.fft
import math

data = pd.read_csv("data.csv")
x1 = data["x[n]"].tolist()
y = data["y[n]"].tolist()

# For denoising y[n]
# We will substitute a value of the signal with the mean of it and its neighbours, 1 on each side
y_denoised = []
y_denoised.append(y[0])
for i in range (1,192):        # n = 193 for the signal y[n], we are excluding the extremities as they'll be included as is
    y_denoised.append(round((y[i-1]+y[i]+y[i+1])/3,4))         # Values of the signal have been rounded to 4 decimal points
y_denoised.append(y[-1])

# Plot of y[n]
xaxis = np.arange(-96,97)
yaxis = np.array(y)
plt.plot(xaxis,yaxis)
plt.show()

# Plot of y[n] after denoising
xaxis = np.arange(-96,97)
yaxis = np.array(y_denoised)
plt.plot(xaxis,yaxis)
plt.show()

# To calculate the Fourier Transform of a 1D signal given by an array x

def dft(x):
    n = np.arange(-1*0.5*(len(x)-1),0.5*(len(x)-1)+1)
    k = n.reshape((len(x), 1))
    e = np.exp(-2j * np.pi * k * n / len(x))
    X = np.dot(e,x)
    return X

y_ft = dft(y_denoised)

# To calculate the frequency for the x-axis
n = np.arange(-1*0.5*(len(y_ft)-1),0.5*(len(y_ft)-1)+1)  # len(y_ft) = 193
T = len(y_ft)/150                                        # We have chosen 150 Hz as the sampling rate
freq = n/T

# Plotting the Fourier Transform of y[n]
plt.title("Discrete Fourier Transform of y[n]")
plt.xlabel("w")
plt.ylabel("Y(e^jw)")
plt.plot(freq,[abs(i) for i in y_ft])
plt.show()

#To calculate the Fourier Transform for the blurring kernel
blur = dft([1/16,4/16,6/16,4/16,1/16])

n = np.arange(-1*0.5*(len(blur)-1),0.5*(len(blur)-1)+1)
T = len(blur)/150                                        # Even the blurring kernel has been sampled at 150 Hz
freq = n/T

plt.title("Discrete Fourier Transform of h[n]")
plt.xlabel("w")
plt.ylabel("H(e^jw)")
plt.plot(freq,[abs(i) for i in blur])
plt.show()
