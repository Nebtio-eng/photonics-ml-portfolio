from src.robustness.robust_analysis import robust_transmission


result = robust_transmission(
    neff=2.4,
    L1=500e-6,
    L2=505e-6,
    wavelength=1550e-9
)

print(result)
