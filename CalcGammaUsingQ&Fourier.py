import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft,rfftfreq
import pandas as pd
# Read data into, change sheet_name for different volumes
data = pd.read_excel("Nitrogen.xlsx", sheet_name="20ml", names=("time","emf"),usecols=(0,1),skiprows=1)

# ------------ Calculate FWHM ------------
maxEMF = max(data.emf) # Variable for maxEMF, highest EMF reached.
        
print("Max EMF is ", maxEMF, "V")
print("Hence, half maximum is when EMF is equal to ", maxEMF/2, "V")

timeHalfEMF = 0
for i in range(0, len(data)):
    if (data.emf[i] - maxEMF/(np.sqrt(2)))**2 < 0.001: # magnitude of difference is within a certain limit.
        timeHalfEMF = data.time[i] # Equal to time at which EMF is half the max.
        
print("This occurs at a time of ", timeHalfEMF, "s")
# ----------------------------------------

# ---------- Fourier Transform -----------
emf=np.array(data.emf)
time=np.array(data.time)
FTfreq = np.fft.rfftfreq(len(time),time[1]-time[0])
FT = np.fft.rfft(emf)
aFT = np.abs(FT)
new_aFT = list() # new list for amplitudes greater than a certain value
for i in range(0, len(aFT)): # loop to find values greater than min
    if aFT[i] > 2.0:
        new_aFT.append(aFT[i])
    else:
        new_aFT.append(0)
# ----------------------------------------

# ----------- Calc Peak Omega ------------
for i in range(len(new_aFT)):
    if new_aFT[i]==max(new_aFT):
        maxAmp = new_aFT[i]
        maxAmpFreq = FTfreq[i]
print("Peak angular frequency is ", maxAmpFreq, "Hz")
# ----------------------------------------

# --------------- Q Factor ---------------
print("Finally, the Q factor is equal to ", maxAmpFreq/timeHalfEMF)
# ----------------------------------------

# ----------------- Plot -----------------
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(data.time,data.emf)
plt.xlabel("Time /s")
plt.ylabel("EMF")
ax2 = fig.add_subplot(212)
ax2.plot(FTfreq, new_aFT/maxAmp)
plt.xlabel("Frequency / Hz")
plt.ylabel("Arbitrary Amplitude")
plt.show()
# ----------------------------------------

# ----------- Calculate Gamma ------------
mass = 0.0991
V = 20 * 10**(-6)     # CHANGE VOLUME WHEN CHANGING THE SHEET
freq = maxAmpFreq
A = (np.pi*(34.16*10**(-3))**2)/4
P = 10**5
Q = maxAmpFreq/timeHalfEMF

gamma = (4*np.pi**2*mass*V*freq**2)/(P*A**2)*1/((1+1/(4*Q**2-1)))
print ("Gamma: ", gamma)
# ----------------------------------------