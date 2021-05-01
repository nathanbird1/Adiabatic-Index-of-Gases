# Plotting EMF as a function of time.

import matplotlib.pyplot as plt ### plotting things
import numpy as np ## one of python's main maths packages
import pandas as pd ## for reading in our data
from scipy.optimize import curve_fit ## for fitting a line to our data
import matplotlib.ticker as ticker ## this one lets us change some parameters in our plots.
#%matplotlib inline 
#%config InlineBackend.figure_format = 'retina' 
from IPython.display import display, Markdown

# Define constants.
N = 100.0 # order of magnitude of 100 turns.
oscAmp = 10**(-2) # Amplitude of oscillations is ~1cm?
coilA = 10**(-4) # Cross-section of coil ~10^(-4)m^2.
gamma = 1.666 # Adiabatic index.
pressure = 10**5 # Pressure of gas, Pa.
pistonA = 10**(-4) # Surface area of piston.
mass = 0.01 # Mass of piston approx 10g?
volume = 100*10**(-6) # Approx 10^(-4)m^2 in 100ml cylinder
dampConst = 1 # Set it to 1, seems to be nice. Not sure what values it usually take in experiment?


omega = np.sqrt((gamma*pressure*(pistonA)**2)/(mass*volume)) # Equation for omega (with no damping) that Jack derived.
omegaDamped = np.sqrt(omega**2-dampConst**2) # Damped frequency.

timeArray = np.arange(0,2,0.001) # Creates array between 0 and 2 secs.
xArray = oscAmp*np.exp(-dampConst*timeArray)*np.cos(omegaDamped*timeArray) # Creates array of displacement values given underdamped harmonic motion.
velocityArray = list() # Velocity is also a list.
emfArray = list() # EMF is also a list.
diffB = 1/(xArray**2) # dB/dt. Assuming inversely prop to square of x. Not used anywhere atm, seesm to screw up graphs.

for i in range(0,len(xArray)-1): # Calculates ∆x/∆t for each value of x.
    velocityArray.append((xArray[i+1]-xArray[i])/(timeArray[i+1]-timeArray[i]))

for i in range(0,len(xArray)-1): # Calculates EMF = -NAv*dB/dt, but taken dB/dt to be 1 as couldnt get it work if included, x100 also to get right order of magnitude.
    emfArray.append(-N*coilA*velocityArray[i]*100)


plt.plot(timeArray, xArray, color = "r", linestyle = ":", label = "Displacement")
timeArray = np.arange(0,1.999,0.001) # Had to adjust range of time values, as otherwise the timeArray has 1 more datapoint than the velocityArray and emfArray.
plt.plot(timeArray, velocityArray, color = "g", linestyle = "--", label = "Velocity")
plt.plot(timeArray, emfArray, color = "b", label = "EMF")
plt.legend(loc="lower right")
plt.show()