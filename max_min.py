from scipy.io import FortranFile
import numpy as np
# import math

lgdenmax = 4  # lg(den upper limit of net electron)
EOSIUSE = int((lgdenmax + 10) * 10 + 1)  # den index
EOSIMAX = 211  # den index
EOSJMAX = 71  # temp index
zbar = 1    # average number of protons per nuclei
abar = 1    # average number of nucleons per nuclei
ye = zbar/abar  # electron mole number

# read the table
fii = FortranFile('helm_table.bdat', 'r')
f = fii.read_reals(dtype=float)
fd = fii.read_reals(dtype=float)
ft = fii.read_reals(dtype=float)
fdd = fii.read_reals(dtype=float)
ftt = fii.read_reals(dtype=float)
fdt = fii.read_reals(dtype=float)
fddt = fii.read_reals(dtype=float)
fdtt = fii.read_reals(dtype=float)
fddtt = fii.read_reals(dtype=float)
# dpdf = fii.read_reals(dtype=float)
# dpdfd = fii.read_reals(dtype=float)
# dpdft = fii.read_reals(dtype=float)
# dpdfdt = fii.read_reals(dtype=float)
# ef = fii.read_reals(dtype=float)
# efd = fii.read_reals(dtype=float)
# eft = fii.read_reals(dtype=float)
# efdt = fii.read_reals(dtype=float)
# xf = fii.read_reals(dtype=float)
# xfd = fii.read_reals(dtype=float)
# xft = fii.read_reals(dtype=float)
# xfdt = fii.read_reals(dtype=float)

# f --  Helmholtz free energy
# fd --  derivative of f wrt density
# ft --  derivative of f wrt temperature
# fdd --  second derivative of f wrt density
# ftt --  second derivative of f wrt temperature
# fdt --  second derivative of f wrt density and temperature
# fddt --  third derivative of f wrt density^2 and temperature
# fdtt --  third derivative of f wrt density and temperature^2 e.g. dF/(dd)(dt^2)
# fddtt --  fourth derivative of f wrt density^2 and temperature^2
# dpdf --  pressure derivative dp/dd
# dpdfd --
# dpdft --
# dpdfdt --
# ef --  electron chemical potential
# efd --
# eft --
# efdt --
# xf --  number density
# xfd --
# xft --
# xfdt --

eos_f = np.transpose(f.reshape(EOSJMAX, EOSIMAX))
eos_fd = np.transpose(fd.reshape(EOSJMAX, EOSIMAX))
eos_ft = np.transpose(ft.reshape(EOSJMAX, EOSIMAX))
eos_fdd = np.transpose(fdd.reshape(EOSJMAX, EOSIMAX))
eos_ftt = np.transpose(ftt.reshape(EOSJMAX, EOSIMAX))
eos_fdt = np.transpose(fdt.reshape(EOSJMAX, EOSIMAX))
eos_fddt = np.transpose(fddt.reshape(EOSJMAX, EOSIMAX))
eos_fdtt = np.transpose(fdtt.reshape(EOSJMAX, EOSIMAX))
eos_fddtt = np.transpose(fddtt.reshape(EOSJMAX, EOSIMAX))
# eos_dpdf = np.transpose(dpdf.reshape(EOSJMAX, EOSIMAX))
# eos_dpdfd = np.transpose(dpdfd.reshape(EOSJMAX, EOSIMAX))
# eos_dpdft = np.transpose(dpdft.reshape(EOSJMAX, EOSIMAX))
# eos_dpdfdt = np.transpose(dpdfdt.reshape(EOSJMAX, EOSIMAX))
# eos_ef = np.transpose(ef.reshape(EOSJMAX, EOSIMAX))
# eos_efd = np.transpose(efd.reshape(EOSJMAX, EOSIMAX))
# eos_eft = np.transpose(eft.reshape(EOSJMAX, EOSIMAX))
# eos_efdt = np.transpose(efdt.reshape(EOSJMAX, EOSIMAX))
# eos_xf = np.transpose(xf.reshape(EOSJMAX, EOSIMAX))
# eos_xfd = np.transpose(xfd.reshape(EOSJMAX, EOSIMAX))
# eos_xft = np.transpose(xft.reshape(EOSJMAX, EOSIMAX))
# eos_xfdt = np.transpose(xfdt.reshape(EOSJMAX, EOSIMAX))
# print(eos_dpdf.shape)


def get_energy(iat, jat):
    denele = 10**(-10.+0.1*iat)
    den = denele / ye
    tem = 10**(4.+0.1*jat)
    # electron - positron e de/dt
    sele = -eos_ft[iat, jat]  # entropy s
    dsepdt = -eos_ftt[iat, jat]  # ds/dt

    eele = eos_f[iat, jat] + tem * sele  # internal energy e
    deepdt = tem * dsepdt  # de/dt

    # ion e de/dt
    kerg = 1.380658e-16
    avo = 6.0221367e23
    kt = kerg * tem
    eion = 1.5e0 * avo * kt / abar
    deiondt = 1.5e0 * avo * kerg / abar

    # rad e de/dt
    c = 2.99792458e10
    ssol = 5.67051e-5
    asol = 4.0e0 * ssol / c
    asoli3 = asol / 3.0e0
    prad = asoli3 * tem * tem * tem * tem
    erad = 3.0e0 * prad / den
    deraddt = 4.0e0 * erad / tem

    # total
    etot = eion + erad + eele
    dedt = deraddt + deiondt + deepdt

    return etot, dedt


emax = []
emin = []
for i in range(EOSIUSE):
    E_T = []
    for j in range(EOSJMAX):
        ereal, dedtreal = get_energy(i, j)
        E_T.append(ereal)
    for k in range(EOSJMAX-1):
        if E_T[k] > E_T[k+1]:
            print('error,den =', 10**(-10.+0.1*i))
            print('tem=', 10**(4.+0.1*k))
    emax.append(max(E_T))
    emin.append(min(E_T))

print(emax)
print(emin)
print('最大值中的最大值', '%.16e' % max(emax))
print('最大值中的最小值', '%.16e' % min(emax))
print('最小值中的最大值', '%.16e' % max(emin))
print('最大值中的最小值', '%.16e' % min(emin))
print(len(emax))
