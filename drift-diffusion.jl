using Plots
using Printf
using Debugger

# function main()
let program_name = "drift-diffusion"
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
    dEc = Vt * log(RNc / ni) # [V] conduction band offset

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
    delta_acc = 1e-3 # preset the tolerance
    dVa = Vt / 10 # [V] voltage step
    # Va_max = 0.625 # [V] maximum applied bias
    Va_max = dVa # [V] maximum applied bias
    Va_max = dVa

    # =============================================================
    # Utility functions
    # =============================================================
    function Ber(x)
        if x > 1e-2
            return x * exp(-x) / (1 .- exp(-x))
        elseif x < -1e-2
            return x / (exp(x) - 1)
        elseif x == 0
            return 1
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
            @assert d[i-1] != 0
            d[i] = b[i] - a[i] * c[i-1] / d[i-1]
        end

        # solution of Lv = f
        v[1] = f[1]
        for i = 2:n_max
            @assert d[i-1] != 0
            v[i] = f[i] - a[i] * v[i-1] / d[i-1]
        end

        # solution of U*x = v
        @assert d[end] != 0
        x[end] = v[end] / d[end]
        for i = n_max-1:-1:1
            @assert d[i] != 0
            x[i] = (v[i] - c[i] * x[i+1]) / d[i]
        end

        return x
    end

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
    dx = min(Ldn, Ldp) / 100 # [cm]

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
    xx = zz .* (1 .+ sign.(zz) .* sqrt.(1 .+ 1.0 ./ zz .^ 2))
    fi = log.(xx)
    n = xx
    p = 1 / xx

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%                                                                 %%
    # %%                Solving the Equilibirium Case                    %%
    # %%                                                                 %%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    @printf("Start solving the equilibrium case...\n")

    # ========================================================================
    # (A) Define the elements of the coefficient matrix for the internal nodes 
    # and initlize the forcing function
    # ========================================================================

    dx2 = dx^2
    n = exp.(fi)
    p = exp.(-fi)
    a = ones(n_max) / dx2
    c = ones(n_max) / dx2
    b = -(2 / dx2 .+ n .+ p)
    f = n .- p .- dop .- fi .* (n .+ p)

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
        @printf("k_iter: %d  delta: %e\n", k_iter, delta)
        # test update in the outer iteration loop
        if delta < delta_acc
            flag_conv = true
        else
            n = exp.(fi)
            p = exp.(-fi)
            b[2:end-1] = -(2 / dx2 .+ n[2:end-1] .+ p[2:end-1])
            f[2:end-1] = n[2:end-1] .- p[2:end-1] .- dop[2:end-1] .- fi[2:end-1] .* (n[2:end-1] .+ p[2:end-1])
        end
    end

    # ========================================================================
    # (D) Calculate the electron and hole densities
    # ========================================================================

    xx = collect(0:n_max-1) * dx * Ldi # [cm] x-axis
    Ec = dEc .- Vt .* fi # [eV], conduction band
    n = exp.(fi)
    p = exp.(-fi)
    ro = -q .* ni .* (n .- p .- dop) # [C/cm^3], total charge density
    el_field1, el_field2 = zeros(n_max), zeros(n_max)
    el_field1[2:end-1] = -(fi[3:end] .- fi[2:end-1]) * Vt / dx / Ldi # [V/cm]
    el_field2[2:end-1] = -(fi[3:end] .- fi[1:end-2]) * Vt / 2 / dx / Ldi # [V/cm]
    nf = n * ni
    pf = p * ni

    # xx_um = xx * 1e4 # [um] x-axis
    # tmp = plot(xx_um, Ec, xlabel="x [um]", ylabel="Potential [eV]", title="Conduction band vs Position - at Equilibrium", label="", linewidth=3)
    # savefig(tmp, "Conduction band vs Position - at Equilibrium.pdf")
    # tmp = plot(xx_um, ro, xlabel="x [um]", ylabel="Charge density [C/cm^3]", title="Total charge density vs Position - at Equilibrium", label="", linewidth=3)
    # savefig(tmp, "Total charge density vs Position - at Equilibrium.pdf")
    # t = plot(xx_um, el_field1, xlabel="x [um]", ylabel="Electric field [V/cm]", title="Electric field vs Position - at Equilibrium(1)", label="", linewidth=3)
    # savefig(t, "Electric field vs Position - at Equilibrium(1).pdf")
    # t = plot(xx_um, el_field2, xlabel="x [um]", ylabel="Electric field [V/cm]", title="Electric field vs Position - at Equilibrium(2)", label="", linewidth=3)
    # savefig(t, "Electric field vs Position - at Equilibrium(2).pdf")
    # t = plot(xx_um, [nf, pf], xlabel="x [um]", ylabel="Density [cm^-3]", title="Electron and hole densities vs Position - at Equilibrium", label=["Electron density" "Hole density"], yaxis=:log, linewidth=3)
    # savefig(t, "Electron and hole densities vs Position - at Equilibrium.pdf")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%                                                                 %%
    # %%                Solving the non-equilibruim case                 %%
    # %%                                                                 %%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @printf("Start solving the non-equilibrium case...\n")

    # ========================================================================
    # Start the main Loop to increment the Anode voltage by Vt, till it
    # reaches the maximum voltage.
    # ========================================================================

    Vas, av_currs = zeros(0), zeros(0)

    v_index, Va = 0, 0
    while Va < Va_max
        Va = Va + dVa
        v_index += 1
        fi[1] += dVa
        append!(Vas, Va)

        # ========================================================================
        # (1) we use constant mobility without field dependancy
        # ========================================================================
        mun = ones(n_max) * mu_n0
        mup = ones(n_max) * mu_p0
        munim1by2, munip1by2, mupim1by2, mupip1by2 = zeros(n_max), zeros(n_max), zeros(n_max), zeros(n_max)
        munim1by2[2:end-1] = (mun[1:end-2] + mun[2:end-1]) / 2
        munip1by2[2:end-1] = (mun[2:end-1] + mun[3:end]) / 2
        mupim1by2[2:end-1] = (mup[1:end-2] + mup[2:end-1]) / 2
        mupip1by2[2:end-1] = (mup[2:end-1] + mup[3:end]) / 2

        flag_conv = false # flag for convergence
        k_iter = 0

        while !flag_conv
            k_iter += 1

            n_old = copy(n)
            p_old = copy(p)
            fi_old = copy(fi)

            # ========================================================================
            # (2) Solve the continuity equation for ELECTRON and HOLE
            # ========================================================================

            # (2.a) Define the elements of the coefficient matrix and initialize
            # the forcing function at the ohmic contacts for continuity equations
            an, cn, bn, fn = zeros(n_max), zeros(n_max), zeros(n_max), zeros(n_max)
            an[1], an[end] = 0, 0
            bn[1], bn[end] = 1, 1
            cn[1], cn[end] = 0, 0
            fn[1], fn[end] = n[1], n[end]

            ap, cp, bp, fp = zeros(n_max), zeros(n_max), zeros(n_max), zeros(n_max)
            ap[1], ap[end] = 0, 0
            bp[1], bp[end] = 1, 1
            cp[1], cp[end] = 0, 0
            fp[1], fp[end] = p[1], p[end]

            # (2.b) coefficients for the continuity equations
            cn[2:end-1] = munip1by2[2:end-1] .* Ber.(fi[3:end] .- fi[2:end-1])
            an[2:end-1] = munim1by2[2:end-1] .* Ber.(fi[1:end-2] .- fi[2:end-1])
            bn[2:end-1] = -(
                munim1by2[2:end-1] .* Ber.(fi[2:end-1] .- fi[1:end-2]) +
                munip1by2[2:end-1] .* Ber.(fi[2:end-1] .- fi[3:end])
            )
            fn = (Ldi^2 * dx2 / Vt) .* (p .* n .- 1) ./ (tau_p0 * (1 .+ n) .+ tau_n0 * (1 .+ p))

            cp[2:end-1] = mupip1by2[2:end-1] .* Ber.(fi[2:end-1] .- fi[3:end])
            ap[2:end-1] = mupim1by2[2:end-1] .* Ber.(fi[2:end-1] .- fi[1:end-2])
            bp[2:end-1] = -(
                mupim1by2[2:end-1] .* Ber.(fi[1:end-2] .- fi[2:end-1]) +
                mupip1by2[2:end-1] .* Ber.(fi[3:end] .- fi[2:end-1])
            )
            fp = (Ldi^2 * dx2 / Vt) .* (p .* n .- 1) ./ (tau_p0 .* (n .+ 1) .+ tau_n0 .* (p .+ 1))

            # (2.c) Solve electron current density equation using LU decomposition method
            n = LuDecomposition(an, bn, cn, fn)
            delta_n = maximum(abs.(n - n_old))
            if isnan(delta_n)
                @printf("NaN detected in `n` at k_iter = %d\n", k_iter)
                exit()
            end

            # (2.d) Solve hole current density equation using LU decomposition method
            p = LuDecomposition(ap, bp, cp, fp)
            delta_p = maximum(abs.(p - p_old))
            if isnan(delta_p)
                @printf("NaN detected in `p` at k_iter = %d\n", k_iter)
                exit()
            end

            # ========================================================================
            # (3) Calculate the potential again using Poisson's equation and check convergence
            # ========================================================================
            a = ones(n_max) / dx2
            c = ones(n_max) / dx2
            b = -(2 / dx2 .+ n .+ p)
            f = n .- p .- dop .- fi .* (n .+ p)

            # (3b) Define the elements of the coefficient matrix and initialize the forcing function at the ohmic contacts
            a[1], a[end] = 0, 0
            b[1], b[end] = 1, 1
            c[1], c[end] = 0, 0
            f[1], f[end] = fi[1], fi[end]

            # (3c) Solve electric potential equation using LU decomposition method
            fi = LuDecomposition(a, b, c, f)
            delta_fi = maximum(abs.(fi - fi_old))
            if isnan(delta_fi)
                @printf("NaN detected in `fi` at k_iter = %d\n", k_iter)
                exit()
            end
            @printf("Va: %f, k_iter: %d, delta_n: %e, delta_p: %e, delta_fi: %e\n", Va, k_iter, delta_n, delta_p, delta_fi)

            if delta_fi < delta_acc
                flag_conv = true
            end
        end

        # ==============================================================================
        # Calculate currents
        # ==============================================================================

        aa_n = q * mun * Vt / dx / Ldi * ni
        aa_p = q * mup * Vt / dx / Ldi * ni

        # (1) Electron current density
        Jnim1by2, Jnip1by2, curr_n = zeros(n_max), zeros(n_max), zeros(n_max)
        Jnim1by2[2:end-1] = (
            n[2:end-1] .* Ber.(fi[2:end-1] .- fi[1:end-2]) .-
            n[1:end-2] .* Ber.(fi[1:end-2] .- fi[2:end-1])
        )
        Jnip1by2[2:end-1] = (
            n[3:end] .* Ber.(fi[3:end] .- fi[2:end-1]) .-
            n[2:end-1] .* Ber.(fi[2:end-1] .- fi[3:end])
        )
        curr_n = (Jnim1by2 .+ Jnip1by2) / 2 .* aa_n

        # (2)Hole current density
        Jpim1by2, Jpip1by2, curr_p = zeros(n_max), zeros(n_max), zeros(n_max)
        Jpim1by2[2:end-1] = (
            p[2:end-1] .* Ber.(fi[1:end-2] .- fi[2:end-1]) .-
            p[1:end-2] .* Ber.(fi[2:end-1] .- fi[1:end-2])
        )
        Jpip1by2[2:end-1] = (
            p[3:end] .* Ber.(fi[2:end-1] .- fi[3:end]) .-
            p[2:end-1] .* Ber.(fi[3:end] .- fi[2:end-1])
        )
        curr_p = (Jpim1by2 .+ Jpip1by2) / 2 .* aa_p

        tot_curr = curr_n .+ curr_p
        tot_curr_sum = sum(tot_curr)

        av_curr = tot_curr_sum / (n_max - 2)
        @printf("Va: %f, av_curr: %e\n", Va, av_curr)
        append!(av_currs, av_curr)
    end

    # ==============================================================================
    # Plot
    # ==============================================================================

    # xx = collect(0:n_max-1) * dx * Ldi
    # cond_band = dEc .- Vt .* fi # [eV], conduction band
    # tot_charge = -q .* ni .* (n .- p .- dop) # [C/cm^3], total charge density
    # el_field1, el_field2 = zeros(n_max), zeros(n_max)
    # el_field1[2:end-1] = -(fi[3:end] .- fi[2:end-1]) * Vt / dx / Ldi # [V/cm]
    # el_field2[2:end-1] = -(fi[3:end] .- fi[1:end-2]) * Vt / 2 / dx / Ldi # [V/cm]
    # efn = Vt * (fi - log.(n)) # [V], electron Fermi level
    # efp = Vt * (fi - log.(p)) # [V], hole Fermi level

    # Plot(xx, cond_band, "Conduction band", "x [nm]", "E [eV]", linewidth=3)
    # Plot(xx, tot_charge, "Total charge density", "x [nm]", "Q [C/cm^3]", linewidth=3)
    # Plot(xx, el_field1, "Electron field(1)", "x [nm]", "E [V/cm]", linewidth=3)
    # Plot(xx, el_field2, "Electron field(2)", "x [nm]", "E [V/cm]", linewidth=3)
    # Plot(xx, n * ni, "Electron density", "x [nm]", "n [cm^-3]", linewidth=3)
    # Plot(xx, p * ni, "Hole density", "x [nm]", "p [cm^-3]", linewidth=3)
    # Plot(xx, efn, "Electron Fermi level", "x [nm]", "E [V]", linewidth=3)
    # Plot(Vas, av_currs, "Average current", "V [V]", "I [A/cm]", linewidth=3)
end