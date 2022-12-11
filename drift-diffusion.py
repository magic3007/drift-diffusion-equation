import math
import matplotlib.pyplot as plt

# =====================================================
# Define fundamental constants and material parameters
# =====================================================
q = 1.602e-19 # [C] electron charge
kb = 1.38e-23 # [J/K] Boltzmann constant
eps = 1.05e-12 # [F/cm] this include the eps = 11.7 for Si
T = 300.0 # [K] temperature
ni = 1.5e10 # [cm^-3] intrinsic carrier concentration
Vt = kb * T / q # [V] thermal voltage
RNc = 2.8e19 # [cm^-3] intrinsic carrier concentration in conduction band
dEc = Vt * math.log(RNc / ni) # [V] conduction band offset

mu_n0 = 1252 # [cm^2/Vs] electron mobility
mu_p0 = 407 # [cm^2/Vs] hole mobility

# =====================================================
# Define doping profile
# =====================================================
N_d = 1e16 # [cm^-3] the density of donor atoms 
N_a = 1e16 # [cm^-3] the density of acceptor atoms

# =====================================================
# Define some material constants
# =====================================================
Ncn = 1.432e17 # [cm^-3] conduction band density of states
Ncp = 2.67e17 # [cm^-3] valence band density of states

tau_n0 = 5e-7 # [s] electron lifetime
tau_p0 = 5e-7 # [s] hole lifetime

# =============================================================
# Define the simulation parameters
# =============================================================
delta_acc = 1e-7 # preset the tolerance
Va_max = 0.66 # [V] maximum applied bias
n_steps = 100 # number of steps


# =============================================================
# Calculate relevant parameters for the simulation
# =============================================================

# the width of depletion region
# Refer to https://www.pveducation.org/pvcdrom/pn-junctions/solving-for-depletion-region
Vbi = Vt * math.log(N_a * N_d / ni / ni) # built-in voltage [V]
W = math.sqrt(2 * eps * Vbi * (N_a + N_d) / q / (N_a * N_d)) # depletion region width [cm]
W_n = W * N_a / (N_a + N_d) # [cm]
W_p = W * N_d / (N_a + N_d) # [cm]

# 
E_p = q * N_d * W_n / eps # [V/cm]

# Debye lengths = sqrt(eps * kb * T / (q^2 * N))
Ldn = math.sqrt(eps * Vt / (q * N_d)) # [cm]
Ldp = math.sqrt(eps * Vt / (q * N_a)) # [cm]
Ldi = math.sqrt(eps * Vt / (q * ni)) # [cm]

# ==============================================================
# Setting the size of the simulation domain based on the 
# analytical results for the width of the depletion region
# ==============================================================
x_max = max(W_n, W_p) * 20 # [cm]

# ==============================================================
# Setting the grid size based on the extrinsic Debye lengths
# ==============================================================
dx = min(Ldn, Ldp) / 20 # [cm]

# ==============================================================
# Calculate the required number of grid points and renormalize dx
# ==============================================================
n_max = int(x_max / dx)
print("Number of grid points: ", n_max)
dx = dx / Ldi # normalized


# =============================================================
# Utility functions
# =============================================================
def Ber(x):
    if x > 1e-4:
        return x * math.exp(-x) / (1 - math.exp(-x))
    elif x < -1e-4:
        return x / (math.exp(x) - 1)
    elif x == 0:
        return 1
    else:
        temp_term = 1
        sum = temp_term
        i = 0
        flag_sum = False
        while not flag_sum:
            i += 1
            temp_term = temp_term * x / (i + 1)
            if sum + temp_term == sum:
                flag_sum = True
            else:
                sum += temp_term
        return 1 / sum

def LuDecomposition(a, b, c, f):
    length = len(a)
    assert len(b) == length
    assert len(c) == length
    assert len(f) == length

    d = [0 for x in range(length)]
    v = [0 for x in range(length)]
    x = [0 for x in range(length)]

    # solution of Ly = b
    d[0] = b[0]
    for i in range(1, length):
        d[i] = b[i] - a[i] * c[i-1] / d[i-1]

    # solution of Lv = f
    v[0] = f[0]
    for i in range(1, length):
        v[i] = f[i] - a[i] * v[i-1] / d[i-1]

    # solution of U*x = v
    x[-1] = v[-1] / d[-1]
    for i in range(length-2, -1, -1):
        x[i] = (v[i] - c[i] * x[i+1]) / d[i]

    return x

def mysign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

# ====================================================================
# Set up the doping C(x) = N_d(x) - N_a(x) that is normalized with ni
# ====================================================================
dop = [0 for x in range(n_max)]
for i in range(n_max):
    if(i > n_max / 2):
        dop[i] = -N_a / ni
    else:
        dop[i] = N_d / ni

# =============================================================
# Utility functions
# =============================================================

def PlotResult(n, p, fi, title):
    xx_um = [1e4 * x * dx * Ldi for x in range(n_max)]# [cm] x-axis
    Ec = [dEc - Vt * fi[i] for i in range(n_max)] # [eV], conduction band
    ro = [-q * ni * (n[i] - p[i] - dop[i]) for i in range(n_max)] # [C/cm^3], total charge density
    el_field1 = [0 for x in range(n_max)] # [V/cm], electric field
    el_field2 = [0 for x in range(n_max)] # [V/cm], electric field
    for i in range(1, n_max-1):
        el_field1[i] = -(fi[i+1] - fi[i]) * Vt / dx / Ldi
        el_field2[i] = -(fi[i+1] - fi[i-1]) * Vt / dx / Ldi / 2
    nf = [ni * x for x in n]
    pf = [ni * x for x in p]

    filename = title + ".pdf"
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    fig.tight_layout(h_pad=3, w_pad=3)
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)
    ax[0][0].plot(xx_um, Ec, linewidth=3)
    ax[0][0].set_xlabel("x [um]")
    ax[0][0].set_ylabel("Potential [eV]")
    ax[0][0].set_title("Conduction band")
    ax[0][0].grid(True)
    
    ax[0][1].plot(xx_um, ro, linewidth=3)
    ax[0][1].set_xlabel("x [um]")
    ax[0][1].set_ylabel("Charge density [C/cm^3]")
    ax[0][1].set_title("Total charge density")
    ax[0][1].grid(True)
    
    ax[1][0].plot(xx_um, el_field1, linewidth=3)
    ax[1][0].set_xlabel("x [um]")
    ax[1][0].set_ylabel("Electric field [V/cm]")
    ax[1][0].set_title("Electric field")
    ax[1][0].grid(True)
    
    ax[1][1].plot(xx_um, nf, linewidth=3, label="Electron density")
    ax[1][1].plot(xx_um, pf, linewidth=3, label="Hole density")
    ax[1][1].set_xlabel("x [um]")
    ax[1][1].set_ylabel("Density [cm^-3]")
    ax[1][1].set_title("Electron and hole densities")
    ax[1][1].set_yscale("log")
    ax[1][1].grid(True)
     
    plt.savefig(filename)
    
# =====================================================================
# Initialize the potential based on the requirement of charge neutrality
# throughout the whole structure
# =====================================================================
zz = list(map(lambda x:0.5* x, dop))
xx = list(map(lambda x: x * (1 + mysign(x) * math.sqrt(1 + 1 / (x * x))), zz))
fi = list(map(lambda x: math.log(x), xx))

print("Start solving the equilibrium case...")

# ========================================================================
# (A) Define the elements of the coefficient matrix for the internal nodes 
# and initlize the forcing function
# ========================================================================
dx2 = dx * dx
n = list(map(math.exp, fi))
p = list(map(lambda x: math.exp(-x), fi))
a = [1 / dx2 for x in range(n_max)]
c = [1 / dx2 for x in range(n_max)]
b = [-(2 / dx2 + n[i] + p[i]) for i in range(n_max)]
f = [n[i] - p[i] - dop[i] - fi[i] * (n[i] + p[i]) for i in range(n_max)]
    
# ========================================================================
# (B) Define the elements of the coefficient matrix and initlize the forcing
# function for the ohmic contacts
# ========================================================================
a[0], a[-1] = 0, 0
b[0], b[-1] = 1, 1
c[0], c[-1] = 0, 0
f[0], f[-1] = fi[0], fi[-1]

# =============================================================================
# (C) Start the iterative procedure for the solution of the linearized Poisson
# equation using the LU decomposition method
# =============================================================================

flag_conv = False # flag for convergence
k_iter = 0
while not flag_conv:
    k_iter += 1
    fi_old = fi.copy()
    fi = LuDecomposition(a, b, c, f)
    n = list(map(math.exp, fi))
    p = list(map(lambda x: math.exp(-x), fi))
    delta = max(map(abs, map(lambda x, y: x - y, fi, fi_old)))
    print(f"iteration: {k_iter}  delta: {delta:.7e}")
    # test update in the outer iteration loop
    if delta < delta_acc:
        flag_conv = True
    else:
        for i in range(1,n_max-1):
            b[i] = -(2 / dx2 + n[i] + p[i])
            f[i] = n[i] - p[i] - dop[i] - fi[i] * (n[i] + p[i])

PlotResult(n,p,fi, "Equilibrium case")
print("Start solving the non-equilibrium case...")

# ========================================================================
# Start the main Loop to increment the Anode voltage by Vt, till it
# reaches the maximum voltage.
# ========================================================================

Vas, av_currs = [], []

Va, v_index = 0, 0
dVa = Va_max / n_steps
while Va < Va_max:
    Va = Va + dVa
    fi[0] += dVa
    v_index += 1

    flag_conv = False # flag for convergence
    k_iter = 0

    while not flag_conv:
        k_iter += 1

        n_old = n.copy()
        p_old = p.copy()
        fi_old = fi.copy()

        # ========================================================================
        # (1) Solve the continuity equation for ELECTRON
        # ========================================================================

        # (1.a) Define the elements of the coefficient matrix and initialize
        # the forcing function at the ohmic contacts for continuity equations
        an = [0 for x in range(n_max)]
        bn = [0 for x in range(n_max)]
        cn = [0 for x in range(n_max)]
        fn = [0 for x in range(n_max)]

        an[0], an[-1] = 0, 0
        bn[0], bn[-1] = 1, 1
        cn[0], cn[-1] = 0, 0
        fn[0], fn[-1] = n[0], n[-1]
        
        # (1.b) coefficients for the continuity equations
        for i in range(1, n_max-1, 1):
            an[i] = Ber(fi[i-1] - fi[i])
            cn[i] = Ber(fi[i+1] - fi[i])
            bn[i] = -(Ber(fi[i] - fi[i-1]) + Ber(fi[i] - fi[i+1]))
            # fn[i] = (Ldi*Ldi * dx2 / Vt) * (p[i] * n[i] - 1) / (tau_p0 * (n[i] + 1) + tau_n0 * (p[i] + 1)) 
            
        # (1.c) Solve electron current density equation using LU decomposition method
        n = LuDecomposition(an, bn, cn, fn)
        delta_n = max(map(abs, map(lambda x, y: x - y, n, n_old)))

        # ========================================================================
        # (2) Solve the continuity equation for HOLE
        # ========================================================================
        # (2.a) Define the elements of the coefficient matrix and initialize
        # the forcing function at the ohmic contacts for continuity equations
        ap = [0 for x in range(n_max)]
        bp = [0 for x in range(n_max)]
        cp = [0 for x in range(n_max)]
        fp = [0 for x in range(n_max)]
        
        ap[0], ap[-1] = 0, 0
        bp[0], bp[-1] = 1, 1
        cp[0], cp[-1] = 0, 0
        fp[0], fp[-1] = n[0], n[-1]

        # (2.b) coefficients for the continuity equations
        for i in range(1, n_max - 1, 1):
            ap[i] =  Ber(fi[i] - fi[i-1])
            cp[i] =  Ber(fi[i] - fi[i+1])
            bp[i] =  -(Ber(fi[i-1] - fi[i]) + Ber(fi[i+1] - fi[i]))
            # fp[i] = (Ldi*Ldi * dx2 / Vt) * (p[i] * n[i] - 1) / (tau_p0 * (n[i] + 1) + tau_n0 * (p[i] + 1)) 
        
        # (2.c) Solve hole current density equation using LU decomposition method
        p = LuDecomposition(ap, bp, cp, fp)
        delta_p = max(map(abs, map(lambda x, y: x - y, p, p_old)))

        # ========================================================================
        # (3) Calculate the potential again using Poisson's equation and check convergence
        # ========================================================================

        # (3.a) Define the elements of the coefficient matrix and initialize the forcing function at the ohmic contacts
        a = [1 / dx2 for x in range(n_max)]
        c = [1 / dx2 for x in range(n_max)]
        b = [-(2 / dx2 + n[i] + p[i]) for i in range(n_max)]
        f = [n[i] - p[i] - dop[i] - fi[i] * (n[i] + p[i]) for i in range(n_max)]

        a[0], a[-1] = 0, 0
        c[0], c[-1] = 0, 0
        b[0], b[-1] = 1, 1
        f[0], f[-1] = fi[0], fi[-1]

        # (3.b) Solve electric potential equation using LU decomposition method
        fi = LuDecomposition(a, b, c, f)
        delta_fi = max(map(abs, map(lambda x, y: x - y, fi, fi_old)))
        
        print(f"Va: {Va:.3f}  k_iter: {k_iter}  delta_n: {delta_n:.7e}  delta_p: {delta_p:.7e}  delta_fi: {delta_fi:.7e}")
        if delta_fi < delta_acc:
            flag_conv = True
        
    
    # ==============================================================================
    # Calculate currents
    # ==============================================================================

    aa_n = q * mu_n0 * Vt / dx / Ldi * ni
    aa_p = q * mu_p0 * Vt / dx / Ldi * ni

    # (1) Electron current density
    curr_n = [0 for x in range(n_max)]
    for i in range(1, n_max-1, 1):
        curr_n[i] = (n[i] * Ber(fi[i] - fi[i-1]) - n[i-1] * Ber(fi[i-1] - fi[i])) * aa_n

    # (2)Hole current density
    curr_p = [0 for x in range(n_max)]
    for i in range(1, n_max-1, 1):
        curr_p[i] = (p[i] * Ber(fi[i+1] - fi[i]) - p[i+1] * Ber(fi[i] - fi[i+1])) * aa_p

    tot_curr = [x+y for x, y in zip(curr_n, curr_p)]
    av_curr = sum(tot_curr) / (n_max - 2)
    Vas.append(Va)
    av_currs.append(av_curr)
    print(f"Va: {Va:.3f}  av_curr: {av_curr:.7e}")
    if v_index % 10 == 0 or Va == Va_max:
        PlotResult(n,p,fi, f"Applied voltage: {Va:.3f} V")
 
plt.plot(Vas, av_currs, linewidth=3)
plt.xlabel("Applied Voltage [V]")
plt.ylabel("Average current [I]")
plt.title("Average current vs. applied voltage")
plt.savefig("Average current vs. applied voltage.pdf")