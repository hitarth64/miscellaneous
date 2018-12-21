# Hitarth Choubisa
# Takes a spectrum file as input - currently added as Spectrum.txt
# Outputs variation of Open circuit voltage, short circuit currents, conversion efficiencies and fill factors as functions of band gaps

import numpy, scipy.interpolate, scipy.integrate, urllib.request, io, tarfile
import matplotlib.pyplot as plt
from numpy import exp
from scipy.optimize import fmin

#defining a few constants
hPlanck = 6.62607004*10**(-34)
c0 = 3*10**8
kB = 1.38*10**(-23)
e = 1.6*10**(-19)
Tcell = 300

data_array = numpy.genfromtxt('Spectrum.txt', delimiter="\t")

wavelength = data_array[:,0] * 10**(-9)
spectrum = data_array[:,1] *10**9

plt.plot(wavelength, spectrum)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral intensity (W/m$^2$/nm)")
plt.title("Light from the sun")
plt.show()

wavelength_min = wavelength[0] 
wavelength_max = wavelength[-1] 
E_min = hPlanck * c0 / wavelength_max
E_max = hPlanck * c0 / wavelength_min


AM15interp = scipy.interpolate.interp1d(wavelength, spectrum)

def SPhotonsPerTEA(Ephoton):
    wavelength = hPlanck * c0 / Ephoton
    return AM15interp(wavelength) * (1 / Ephoton) * (hPlanck * c0 / Ephoton**2)

def solar_photons_above_gap(Egap):
    return scipy.integrate.quad(SPhotonsPerTEA, Egap, E_max, full_output=1)[0]
	
PowerPerTEA = lambda E : E * SPhotonsPerTEA(E)
solar_constant = scipy.integrate.quad(PowerPerTEA,E_min,E_max, full_output=1)[0]
print(solar_constant)

Egap_list = numpy.linspace(0.5*e, 4*e, num=100)
y_values = numpy.array([solar_photons_above_gap(E) for E in Egap_list])
plt.plot(numpy.linspace(0.5,4,num=100),y_values)
plt.xlabel("Bandgap energy (eV)")
plt.ylabel("Photons above bandgap");
plt.show()

def RR0(Egap):
    integrand = lambda E : E**2 / (exp(E / (kB * Tcell)) - 1)
    integral = scipy.integrate.quad(integrand, Egap, E_max, full_output=1)[0]
    return ((2 * numpy.pi) / (c0**2 * hPlanck**3)) * integral

# Returns curret density	
def current_density(V, Egap):
    return e * (solar_photons_above_gap(Egap) - RR0(Egap) * exp(e * V / (kB * Tcell)))

# Returns short circuit current
def JSC(Egap):
    return current_density(0, Egap)

# Returns open-circuit voltage
def VOC(Egap):
    return (kB * Tcell / e) * numpy.log(solar_photons_above_gap(Egap) / RR0(Egap))

# Plotting short-circuit current
JSC_list = numpy.array([JSC(E) for E in Egap_list])
plt.plot(Egap_list / e , JSC_list)
plt.xlabel("Bandgap energy (eV)")
plt.ylabel("Ideal short-circuit current ")
plt.title("Ideal short-circuit current as a function of bandgap")
plt.show()

#Plotting Open-circuit voltage
VOC_list = numpy.array([VOC(E) for E in Egap_list])
plt.plot(Egap_list / e , VOC_list )
plt.xlabel("Bandgap energy (eV)")
plt.ylabel("Ideal open-circuit voltage (V)")
plt.title("Ideal open-circuit voltage as a function of bandgap")
plt.show()


def fmax(func_to_maximize, initial_guess=0):
    """return the x that maximizes func_to_maximize(x)"""
    func_to_minimize = lambda x : -func_to_maximize(x)
    return fmin(func_to_minimize, initial_guess, disp=False)[0]

def V_mpp(Egap):
    """ voltage at max power point """
    return fmax(lambda V : V * current_density(V, Egap))

def J_mpp(Egap):
    """ current at max power point """
    return current_density(V_mpp(Egap), Egap)

def max_power(Egap):
    V = V_mpp(Egap)
    return V * current_density(V, Egap)

def max_efficiency(Egap):
    return max_power(Egap) / solar_constant
	
# Plotting Efficiency
Egap_list = numpy.linspace(0.5 * e, 4 * e, num=30)
eff_list = numpy.array([max_efficiency(E) for E in Egap_list])

index_min = numpy.argmax(eff_list)
print("Operating Temperature (in K)", Tcell)
print("Optimal Bandgap: ", Egap_list[index_min] / e)

plt.plot(Egap_list / e , 100 * eff_list)
plt.xlabel("Bandgap energy (eV)")
plt.ylabel("Max efficiency (%)")
plt.title("SQ efficiency limit as a function of bandgap")
plt.show()

# Returns fill factor
def fill_factor(Egap):
    return max_power(Egap) / (JSC(Egap) * VOC(Egap))

# Plotting Fill factor
FF_list = numpy.array([fill_factor(E) for E in Egap_list])
plt.plot(Egap_list / e , FF_list)
plt.xlabel("Bandgap energy (eV)")
plt.ylabel("Ideal fill factor")
plt.title("Ideal fill factor as a function of bandgap")
plt.show()

