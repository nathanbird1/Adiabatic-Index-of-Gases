import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

data = pd.read_csv("air 100ml-f.csv",names=("time","emf"),skiprows=2)
#data = pd.read_excel("Nitrogen.xlsx", sheet_name="100ml", names=("time","emf"),usecols=(0,1),skiprows=1)

clean_time = []
clean_emf = []

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(data.time, data.emf)
plt.xlabel('Time / s')
plt.ylabel('EMF / V')
plt.tick_params(direction='in',      
                length=7,            
                bottom='on',         
                left='on',
                top='on',
                right='on')
plt.plot(data.time,data.emf)

lower = float(input("Where does the data start? "))
upper = float(input("Where does the data end? "))

for i in range(0,len(data.time)):
    if data.time[i]>=lower and data.time[i]<=upper:
        clean_time.append(data.time[i])
        clean_emf.append(data.emf[i])
m = 0.10668

#def curve(t,A,y,f,p):    
 #   return A*np.exp(-y*t)*((2*np.pi*np.cos(2*np.pi*f*t+p))-y*np.sin(2*np.pi*f*t+p))

time = np.arange(lower,upper,0.0001)

def fit(time, A, beta, omega_star, phi):    
    exp_term = A*np.exp(-beta*time)
    cos = np.cos(omega_star*time + phi)

    return exp_term*(cos)

popt, pcov = curve_fit(fit,clean_time,clean_emf)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.tick_params(direction='in',
                length=7,
                bottom='on',
                left='on',
                top='on',
                right='on')
plt.xlabel('Time / s')
plt.ylabel('EMF / V')
plt.plot(clean_time,clean_emf)
plt.plot(time,fit(time,popt[0],popt[1],popt[2],popt[3]),linestyle = '-', color = 'black')
plt.tick_params(direction='in',top='on',bottom='on',left='on',right='on')
plt.rcParams.update({'font.size':15})

print("Amplitude = {:f} +- {:f}".format(popt[0],(pcov[0,0]**0.5)))
print("Damping constant (beta) = {:.3f} +- {:.3f}".format(popt[1],(pcov[1,1]**0.5)))
print("Angular Frequency = {:.3f} +- {:.3f}".format(popt[2],(pcov[2,2]**0.5)))

undamped = np.sqrt(popt[2]**2+(popt[1])**2)
undamped_err = np.sqrt(((popt[2]**2+popt[1]**2)**-0.5*popt[2]*(pcov[2,2])**0.5)**2
                       +((popt[2]**2+popt[1]**2)**-0.5*popt[1]*(pcov[1,1])**0.5)**2)

print("Undamped angular frequency = {:.3f} +- {:.3f}".format(undamped,undamped_err))