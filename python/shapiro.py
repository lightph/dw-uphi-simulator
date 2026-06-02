import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import j1

C, L = np.genfromtxt("output/forzado/shapiro/run1.csv", delimiter=",", unpack=True)


# 1. Define the Bessel J1 model function
# Parameters: a (amplitude), b (frequency scale), c (phase shift), d (vertical offset)
def bessel_func(x, a, b):
    return np.abs(a * j1(b * x))


# 2. Fit the data
# Note: You may need to provide an initial guess (p0=[...]) if the fit doesn't converge automatically.
popt, pcov = curve_fit(bessel_func, C, L, p0=[1.0, 1.0])
# ... (Previous imports and curve_fit line) ...

# 1. Calculate Parameter Errors and Fit Quality
# The diagonal of pcov contains the variance of the parameter estimates
perr = np.sqrt(np.diag(pcov))

# Calculate R-squared (Coefficient of Determination)
residuals = L - bessel_func(C, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((L - np.mean(L)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Print text report to console
print(f"Fit Quality (R^2): {r_squared:.5f}")
print("Parameters:")
params = ["a (Amp)", "b (Freq)", "c (Phase)", "d (Offset)"]
for param, val, err in zip(params, popt, perr):
    print(f"  {param}: {val:.5e} ± {err:.5e}")

# 2. Update Plot with Info Box
fig, ax = plt.subplots(1, 1, tight_layout=True)

ax.scatter(C, L, s=1.0, label="Data")

x_fit = np.linspace(C.min(), C.max(), 1000)
y_fit = bessel_func(x_fit, *popt)
ax.plot(x_fit, y_fit, color="red", label="Bessel J1 Fit")

# Create a text string for the plot legend/box
# Uses LaTeX formatting for Greek letters if Matplotlib supports it
textstr = "\n".join(
    (
        r"$R^2=%.4f$" % (r_squared,),
        r"$a=%.3g \pm %.3g$" % (popt[0], perr[0]),
        r"$b=%.3g \pm %.3g$" % (popt[1], perr[1]),
    )
)

# Place a text box in the upper left (change x, y for placement)
props = dict(boxstyle="round", facecolor="white", alpha=0.8)
ax.text(
    0.05,
    0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=props,
)

ax.legend(loc="upper right")
plt.show()
