using Plots
using Printf

# =====================================================
# Define fundamental constants and material parameters
# =====================================================
q = 1.60217662e-19 # [C] electron charge
kb = 1.38064852e-23 # [J/K] Boltzmann constant
eps = 1.05e-12 # [F/cm] this include the eps = 11.7 for Si
T = 300.0 # [K] temperature
ni = 1.5e10 # [cm^-3] intrinsic carrier concentration
Vt = kb * T / q # [V] thermal voltage
RNc = 2.8e20 # [cm^-3] intrinsic carrier concentration in conduction band
dEc = Vt * log(RNc / ni) # [V] conduction band offset

# =====================================================
# Define doping profile
# =====================================================
N_d = 1e16 # [cm^-3] the density of donor atoms 
N_a = 1e16 # [cm^-3] the density of acceptor atoms
dVa_ = 1e-1 # [V] voltage step
Va_max = 1 # [V] maximum applied bias

# =====================================================
# Define some material constants
# =====================================================
Ncn = 1.432e17 # [cm^-3] conduction band density of states
Ncp = 2.67e17 # [cm^-3] valence band density of states

rmu_1n = 88 # [cm^2/Vs] mobility of electrons
rmu_2n = 1252 # [cm^2/Vs] mobility of electrons
rmu_1p = 54 # [cm^2/Vs] mobility of holes
rmu_2p = 407 # [cm^2/Vs] mobility of holes

tau_n0 = 5e-7 # [s] electron lifetime
tau_p0 = 5e-7 # [s] hole lifetime

Nsrh_n = 5e16 # [cm^-3] electron SRH recombination rate
Nsrh_p = 5e16 # [cm^-3] hole SRH recombination rate

# =============================================================
# Define the simulation parameters
# =============================================================
delta_acc = 1e-10 # preset the tolerance

# =============================================================
# Calculate relevant parameters for the simulation
# =============================================================

# the width of depletion region
# Refer to https://www.pveducation.org/pvcdrom/pn-junctions/solving-for-depletion-region
Vbi = Vt * log(N_a * N_d / ni^2) # built-in voltage [V]
W = sqrt(2 * eps * Vbi * (N_a + N_d) / q / (N_a * N_d)) # depletion region width [cm]
W_n = W * N_a / (N_a + N_d) # [cm]
W_p = W * N_d / (N_a + N_d) # [cm]

# 
E_p = q * N_d * W_n / eps # [V/cm]

# Debye lengths = sqrt(eps * kb * T / (q^2 * N))
Ldn = sqrt(eps * Vt / (q * N_d)) # [cm]
Ldp = sqrt(eps * Vt / (q * N_a)) # [cm]
Ldi = sqrt(eps * Vt / (q * ni)) # [cm]

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
n_max = round(Int, x_max / dx)
println("Number of grid points: ", n_max)
dx = dx / Ldi # normalized

# ====================================================================
# Set up the doping C(x) = N_d(x) - N_a(x) that is normalized with ni
# ====================================================================
# create a array, the first half is -Na, the second half is Nd
dop = zeros(n_max)
dop[1:round(Int, n_max / 2)] .= -N_a
dop[round(Int, n_max / 2)+1:end] .= N_d
dop = dop / ni

# =====================================================================
# Initialize the potential based on the requirement of charge neutrality
# throughout the whole structure
# =====================================================================
zz = 0.5 * dop
fi = log.(zz .* (1 .+ sign.(zz) .* sqrt.(1 .+ 1.0 ./ zz .^ 2)))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%                                                                 %%
# %%                Solving the Equilibirium Case                    %%
# %%                                                                 %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Ber(x)
    if x > 1e-2
        return x * exp(-x) / (1 .- exp(-x))
    elseif x < -1e-2
        return x / (exp(x) - 1)
    else
        temp_term = 1
        sum = temp_term
        i = 0
        flag_sum = false
        while !flag_sum
            i += 1
            temp_term = temp_term * x / (i + 1)
            if sum + temp_term == sum
                flag_sum = true
            else
                sum += temp_term
            end
        end
        return sum
    end
end

function LuDecomposition(a, b, c, f)
    len = length(a)
    @assert length(b) == len
    @assert length(c) == len
    @assert length(f) == len

    d, v, x = zeros(len), zeros(len), zeros(len)
    # solution of Ly = b
    d[1] = b[1]
    for i = 2:n_max
        d[i] = b[i] - a[i] * c[i-1] / d[i-1]
    end

    # solution of Lv = f
    v[1] = f[1]
    for i = 2:n_max
        v[i] = f[i] - a[i] * v[i-1] / d[i-1]
    end

    # solution of U*x = v
    x[end] = v[end] / d[end]
    for i = n_max-1:-1:1
        x[i] = (v[i] - c[i] * x[i+1]) / d[i]
    end

    return x
end

# ========================================================================
# (A) Define the elements of the coefficient matrix for the internal nodes 
# and initlize the forcing function
# ========================================================================

dx2 = dx^2
a = ones(n_max) / dx2
c = ones(n_max) / dx2
b = -(2 / dx2 .+ exp.(fi) .+ exp.(-fi))
f = exp.(fi) .- exp.(-fi) .- dop .- fi .* (exp.(fi) .+ exp.(-fi))

# ========================================================================
# (B) Define the elements of the coefficient matrix and initlize the forcing
# function for the ohmic contacts
# ========================================================================
a[1], a[end] = 0, 0
c[1], c[end] = 0, 0
b[1], b[end] = 1, 1
f[1], f[end] = fi[1], fi[end]

# =============================================================================
# (C) Start the iterative procedure for the solution of the linearized Poisson
# equation using the LU decomposition method
# =============================================================================


flag_conv = false # flag for convergence
k_iter = 0
while !flag_conv
    k_iter += 1
    fi_old = copy(fi)
    fi = LuDecomposition(a, b, c, f)
    delta = maximum(abs.(fi - fi_old))
    @printf("k_iter: %d  delta: %f\n", k_iter, delta)
    # test update in the outer iteration loop
    if delta < delta_acc
        flag_conv = true
    else
        b[2:end-1] = -(2 / dx2 .+ exp.(fi[2:end-1]) .+ exp.(-fi[2:end-1]))
        f[2:end-1] = exp.(fi[2:end-1]) .- exp.(-fi[2:end-1]) .- dop[2:end-1] .- fi[2:end-1] .* (exp.(fi[2:end-1]) .+ exp.(-fi[2:end-1]))
    end
end

# ========================================================================
# (D) Calculate the electron and hole densities
# ========================================================================

xx = collect(0:n_max-1) * dx * Ldi # [cm] x-axis
cond_band0 = dEc .- Vt .* fi # [eV], conduction band
tot_charge0 = -q .* ni .* (exp.(fi) .- exp.(-fi) .- dop) # [C/cm^3], total charge density
el_field1, el_field2 = zeros(n_max), zeros(n_max)
el_field1[2:end-1] = -(fi[3:end] .- fi[2:end-1]) * Vt / dx / Ldi # [V/cm]
el_field2[2:end-1] = -(fi[3:end] .- fi[1:end-2]) * Vt / 2 / dx / Ldi # [V/cm]
n = exp.(fi)
p = exp.(-fi)

plot(xx, cond_band0, label="Conduction band0", linewidth=3)
plot(xx, tot_charge0, label="Total charge density0", linewidth=3)
plot(xx, el_field1, label="Electric field0 (1)", linewidth=3)
plot(xx, el_field2, label="Electric field0 (2)", linewidth=3)
plot(xx, n * ni, label="Electron density0", linewidth=3)
plot(xx, p * ni, label="Hole density0", linewidth=3)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%                                                                 %%
# %%                Solving the non-equilibruim case                 %%
# %%                                                                 %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dVa = dVa_ / Vt # normalized
Va = 0
@assert Va_max > 0

Vas = collect(0:dVa:Va_max)
av_currs = zeros(length(Vas))

println("Convergence of the Gummel cycles")


while Va < Va_max / Vt
    Va = Va + dVa
    fi[1] += dVa

    flag_conv = false # flag for convergence
    k_iter = 0

    while !flag_conv
        k_iter += 1

        # (1) solution of electron current density equation

        # (1a) Define the elements of the coefficient matrix and initialize
        # the forcing function 
        dx2 = dx^2 * Ldi^2
        NN = abs.(dop)
        denom = 1 + NN / Ncn
        rmu_n = rmu_1n + rmu_2n / denom
        y = 175 / 300
        yy = (exp(y) - exp(-y)) / (exp(y) + exp(-y))
        vsat_n = 1.38e7 * sqrt(yy)
        beat_n = 2
        f_plus, f_min = zeros(n_max), zeros(n_max)
        f_plus[2:end-1] = abs(fi[2:end-1] .- fi[3:end]) / dx * Vt / Ldi
        f_min[2:end-1] = abs(fi[1:end-2] .- fi[2:end-1]) / dx * Vt / Ldi

        denom = rmu_n * f_plus / vsat_n
        denom = 1 + denom .^ beat_n
        rmu_plus = rmu_n * (1 / denom) .^ (1 / beat_n)

        denom = rmu_n * f_min / vsat_n
        denom = 1 + denom .^ beat_n
        rmu_min = rmu_n * (1 / denom) .^ (1 / beat_n)

        diff_min = Vt * rmu_min
        diff_plus = Vt * rmu_plus

        an, cn, bn = zeros(n_max), zeros(n_max), zeros(n_max)
        an[2:end-1] = diff_min .* Ber(fi[1:end-2] .- fi[2:end-1])
        cn[2:end-1] = diff_plus .* Ber(fi[3:end] .- fi[2:end-1])
        bn = -(
            diff_min .* Ber(fi[2:end-1] - fi[1:end-2]) +
            dif_plus .* Ber(fi[2:end-1] - fi[3:end])
        )
        tau_n = tau_n0 / (1 + NN / Nsrh_n)
        tau_p = tau_p0 / (1 + NN / Nsrh_p)
        nrum = n .* p - 1
        denom = tau_n * (p + 1) .+ tau_p * (n + 1)
        fn = rnum / denom * dx2

        # (1b) Define the elements of the coefficient matrix and initialize
        # the forcing function at the ohmic contacts
        an[1], an[end] = 0, 0
        cn[1], cn[end] = 0, 0
        bn[1], bn[end] = 1, 1
        fn[1], fn[end] = n[1], n[end]

        # (1c) Solve electron current density equation using LU decomposition method
        n_old = copy(n)
        n = LuDecomposition(an, bn, cn, fn)

        # (2) solution of hole current density equation
        # (2a) Define the elements of the coefficient matrix and initialize the forcing function
        dx2 = dx^2 * Ldi^2
        NN = abs.(dop)
        denom = 1 + NN / Ncp
        rmu_p = rmu_1p + rmu_2p / denom # low-field hole mobility
        y = 312 / 300
        yy = (exp(y) - exp(-y)) / (exp(y) + exp(-y))
        vsat_p = 9.05e6 * sqrt(yy)
        beta_p = 1
        f_plus, f_min = zeros(n_max), zeros(n_max)
        f_plus[2:end-1] = abs(fi[2:end-1] .- fi[3:end]) / dx * Vt / Ldi
        f_min[2:end-1] = abs(fi[1:end-2] .- fi[2:end-1]) / dx * Vt / Ldi
        denom = rmu_p * f_plus / vsat_p
        denom = 1 + denom .^ beta_p
        rmu_plus = rmu_p * (1 / denom) .^ (1 / beta_p)
        denom = rmu_p * f_min / vsat_p
        denom = 1 + denom .^ beta_p
        rmu_min = rmu_p * (1 / denom) .^ (1 / beta_p)
        diff_min = Vt * rmu_min
        diff_plus = Vt * rmu_plus
        ap, cp, bp = zeros(n_max), zeros(n_max), zeros(n_max)
        ap[2:end-1] = diff_min .* Ber(fi[2:end-1] .- fi[1:end-2])
        cp[2:end-1] = diff_plus .* Ber(fi[2:end-1] .- fi[3:end])
        bp = -(
            diff_min .* Ber(fi[1:end-2] .- fi[2:end-1]) +
            diff_plus .* Ber(fi[3:end] .- fi[2:end-1])
        )
        tau_n = tau_n0 / (1 + NN / Nsrh_n)
        tau_p = tau_p0 / (1 + NN / Nsrh_p)
        nrum = n .* p - 1
        denom = tau_n * (p + 1) .+ tau_p * (n + 1)
        fp = rnum / denom * dx2

        # (2b) Define the elements of the coefficient matrix and initialize the forcing function at the ohmic contacts
        ap[1], ap[end] = 0, 0
        cp[1], cp[end] = 0, 0
        bp[1], bp[end] = 1, 1
        fp[1], fp[end] = p[1], p[end]

        # (2c) Solve hole current density equation using LU decomposition method
        p_old = copy(p)
        p = LuDecomposition(ap, bp, cp, fp)

        # (3) solution of electric potential equation
        # (3a) Define the elements of the coefficient matrix and initialize the forcing function
        dx2 = dx^2
        a = ones(n_max) / dx2
        c = ones(n_max) / dx2
        b = -(2 / dx2 + (n .+ p))
        f = n .- p .- dop .- fi .* (n .+ p)

        # (3b) Define the elements of the coefficient matrix and initialize the forcing function at the ohmic contacts
        a[1], a[end] = 0, 0
        c[1], c[end] = 0, 0
        b[1], b[end] = 1, 1
        f[1], f[end] = fi[1], fi[end]

        # (3c) Solve electric potential equation using LU decomposition method
        fi_old = copy(fi)
        fi = LuDecomposition(a, b, c, f)

        delta = maximum(abs.(fi - fi_old))
        @printf("Va: %f, k_iter: %d, delta: %e\n", Va, k_iter, delta)
        if delta < delta_acc
            flag_conv = true
        else
            b[2:end-1] = -(2 / dx2 + (n[2:end-1] .+ p[2:end-1]))
            f[2:end-1] = n[2:end-1] .- p[2:end-1] .- dop[2:end-1] .- fi[2:end-1] .* (n[2:end-1] .+ p[2:end-1])
        end
    end

    # ==============================================================================
    # Calculate currents
    # ==============================================================================

    aa = q * ni * Vt / dx / Ldi

    # ==============================================================================
    # Electron current density
    # ==============================================================================

    NN = abs.(dop)
    denom = 1 + NN / Ncp
    rmu_n = rmu_1n + rmu_2n / denom # low-field electron mobility
    y = 175 / 300
    yy = (exp(y) - exp(-y)) / (exp(y) + exp(-y))
    vsat_n = 1.38e7 * sqrt(yy)
    beta_n = 2
    f_plus, f_min, curr_n = zeros(n_max), zeros(n_max), zeros(n_max)
    f_plus[2:end-1] = abs(fi[2:end-1] .- fi[3:end]) / dx * Vt / Ldi
    denom = rmu_n * f_plus / vsat_n
    denom = 1 + denom .^ beta_n
    rmu_plus_n = rmu_n * (1 / denom) .^ (1 / beta_n)
    curr_n[2:end-1] = rmu_plus_n * (n[3:end] * Ber.(fi[3:end], fi[2:end-1]) .- n[2:end-1] * Ber.(fi[2:end-1], fi[3:end]))

    # ==============================================================================
    # Hole current density
    # ==============================================================================

    NN = abs.(dop)
    denom = 1 + NN / Ncp
    rmu_p = rmu_1p + rmu_2p / denom # low-field hole mobility
    y = 312 / 300
    yy = (exp(y) - exp(-y)) / (exp(y) + exp(-y))
    vsat_p = 9.05e6 * sqrt(yy)
    beta_p = 1
    f_plus, f_min, curr_p = zeros(n_max), zeros(n_max), zeros(n_max)
    f_plus[2:end-1] = abs(fi[2:end-1] .- fi[3:end]) / dx * Vt / Ldi
    denom = rmu_p * f_plus / vsat_p
    denom = 1 + denom .^ beta_p
    rmu_plus_p = rmu_p * (1 / denom) .^ (1 / beta_p)
    curr_p[2:end-1] = rmu_plus_p * (p[2:end-1] * Ber.(fi[3:end], fi[2:end-1]) .- p[3:end] * Ber.(fi[2:end-1], fi[3:end]))

    curr_n *= aa
    curr_p *= aa
    tot_curr = curr_n .+ curr_p
    tot_curr_sum = sum(tot_curr)

    av_curr = tot_curr_sum / (n_max - 2)
    @printf("Va: %f, av_curr: %e\n", Va * Vt, av_curr)
    av_currs.push!(av_curr)

end

# ==============================================================================
# Plot
# ==============================================================================

xx = collect(0:n_max-1) * dx * Ldi
cond_band = dEc .- Vt .* fi # [eV], conduction band
tot_charge = -q .* ni .* (n .- p .- dop) # [C/cm^3], total charge density
el_field1, el_field2 = zeros(n_max), zeros(n_max)
el_field1[2:end-1] = -(fi[3:end] .- fi[2:end-1]) * Vt / dx / Ldi # [V/cm]
el_field2[2:end-1] = -(fi[3:end] .- fi[1:end-2]) * Vt / 2 / dx / Ldi # [V/cm]
efn = Vt * (fi - log.(n)) # [V], electron Fermi level
efp = Vt * (fi - log.(p)) # [V], hole Fermi level

# Plot(xx, cond_band, "Conduction band", "x [nm]", "E [eV]", linewidth=3)
# Plot(xx, tot_charge, "Total charge density", "x [nm]", "Q [C/cm^3]", linewidth=3)
# Plot(xx, el_field1, "Electron field(1)", "x [nm]", "E [V/cm]", linewidth=3)
# Plot(xx, el_field2, "Electron field(2)", "x [nm]", "E [V/cm]", linewidth=3)
# Plot(xx, n * ni, "Electron density", "x [nm]", "n [cm^-3]", linewidth=3)
# Plot(xx, p * ni, "Hole density", "x [nm]", "p [cm^-3]", linewidth=3)
# Plot(xx, efn, "Electron Fermi level", "x [nm]", "E [V]", linewidth=3)
# Plot(xx, efp, "Hole Fermi level", "x [nm]", "E [V]", linewidth=3)

Plot(Vas, av_currs, "Average current", "V [V]", "I [A/cm]", linewidth=3)