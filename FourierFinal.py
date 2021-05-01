import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft,rfftfreq
import pandas as pd

#data = pd.read_excel("Nitrogen.xlsx", sheet_name="100ml", names=("time","emf"),usecols=(0,1),skiprows=1)
#data = pd.read_excel("Nitrogen.xlsx", sheet_name="80ml", names=("time","emf"),usecols=(0,1),skiprows=1)
#data = pd.read_excel("Nitrogen.xlsx", sheet_name="60ml", names=("time","emf"),usecols=(0,1),skiprows=1)
#data = pd.read_excel("Nitrogen.xlsx", sheet_name="40ml", names=("time","emf"),usecols=(0,1),skiprows=1)
data = pd.read_excel("Nitrogen.xlsx", sheet_name="20ml", names=("time","emf"),usecols=(0,1),skiprows=1)

#power = rfft(data100.emf)
#freq = rfftfreq(N, 1/samplerate)

#plt.plot(data.time, data.emf)
#plt.show()

#plt.plot(freq, np.abs(power))
#plt.show()

startTime = -0.208
stopTime = 2.188
sampleRate = 1.
emf=np.array(data.emf)
time=np.array(data.time)
FTfreq = np.fft.rfftfreq(len(time),time[1]-time[0])
FT = np.fft.rfft(emf)
aFT = np.abs(FT)
for i in range(len(aFT)):
    if aFT[i]<3:
        aFT[i]=0
#iFT = np.fft.irfft(aFT)
#newemf=np.array(iFT)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(data.time,data.emf)
plt.title('Signal')
ax2 = fig.add_subplot(212)
ax2.plot(FTfreq, aFT)
#ax2.plot(data100.time,iFT)
plt.title('FourierTransform')
plt.show()
for i in range(len(aFT)):
    if aFT[i]==max(aFT):
        print(FTfreq[i])