################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script creates functions specific to the Monte Carlo simulation with
# β (present bias) estimation. These functions handle data simulation from a
# known DGP and MC-specific versions of estimation functions that work with
# simulated (in-memory) data rather than CSV files.
#
# Changes from 01_MC_Simulation_Functions.jl:
#   - objective_mc() takes 14-element θ_vec (13 structural + β)
#   - Extracts β = θ_vec[14], passes θ_vec[1:13] to get_flow_utility
#   - Passes β and global Π_tya to solve_vfi
################################################################################


#############################
# Logging
#############################

# Uses log_io and log_msg() from 01_Functions_Beta.jl (included before this file).

# Global evaluation counter (reset before each replication)
eval_count = 0


#############################
# Data Simulation
#############################

"""
Simulate household choices from a known DGP using real observed data.

Design-based MC simulation: conditions on real observables (prices, TYA,
panel structure) and only simulates choices from the model. This provides
realistic cross-sectional price variation needed to identify all parameters.

Two-pass approach (matches actual estimation):
  Pass 1: Simulate choices starting from a₀ = 0 to get preliminary choices
  Pass 2: Use preliminary choices to compute a₀ via fixed-point iteration,
           then re-simulate choices from the corrected a₀

Returns:
- y_sim:             Vector{Int} of chosen alternatives (length N_obs)
- tya_state_sim:     Vector{Int} of TYA state indices (length N_obs)
- p_continuous_sim:  Matrix{Float64} of continuous prices (N_obs × 2)
- hh_codes_sim:      Vector{Int} of household identifiers (length N_obs)
"""
function simulate_data(
    V_decision_true::Array{Float64, 4},
    ψ::Real,
    N_J::Integer,
    N_P::Integer,
    A::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    n::AbstractVector{<:Real},
    real_p_continuous::Matrix{Float64},
    real_tya_state::AbstractVector{<:Integer},
    real_hh_codes::AbstractVector{<:Integer}
)

    N_obs = length(real_hh_codes)

    # Output arrays (prices and TYA are taken from real data)
    tya_state_sim    = copy(real_tya_state)
    p_continuous_sim = copy(real_p_continuous)
    hh_codes_sim     = copy(real_hh_codes)

    # Group observations by household
    hh_obs = Dict{Int, Vector{Int}}()
    for i in 1:N_obs
        hh = real_hh_codes[i]
        if !haskey(hh_obs, hh)
            hh_obs[hh] = Int[]
        end
        push!(hh_obs[hh], i)
    end

    # --- Pass 1: simulate choices from a₀ = 0 ---
    y_prelim = Vector{Int}(undef, N_obs)
    for (hh, obs_indices) in hh_obs
        a_h = 0.0
        for i in obs_indices
            v_interp = interpolate_v_choice(
                V_decision_true, real_tya_state[i], a_h,
                real_p_continuous[i, 1], real_p_continuous[i, 2],
                N_J, N_P, A, P
            )
            v_max = maximum(v_interp)
            exp_v = exp.(v_interp .- v_max)
            probs = exp_v ./ sum(exp_v)
            j = categorical_sample(probs)
            y_prelim[i] = j
            # a_h is ã (normalized addiction); n[j] is n_std (standardized nicotine)
            a_h = clamp((1 - ψ) * a_h + ψ * n[j], A[1], A[end])
        end
    end

    # --- Compute a₀ via fixed-point iteration using preliminary choices ---
    a0, _ = get_initial_addiction_stock_mc(ψ, A, n, y_prelim, real_hh_codes)

    # --- Pass 2: re-simulate choices from corrected a₀ ---
    y_sim = Vector{Int}(undef, N_obs)
    for (hh, obs_indices) in hh_obs
        a_h = a0[hh]
        for i in obs_indices
            v_interp = interpolate_v_choice(
                V_decision_true, real_tya_state[i], a_h,
                real_p_continuous[i, 1], real_p_continuous[i, 2],
                N_J, N_P, A, P
            )
            v_max = maximum(v_interp)
            exp_v = exp.(v_interp .- v_max)
            probs = exp_v ./ sum(exp_v)
            j = categorical_sample(probs)
            y_sim[i] = j
            # a_h is ã (normalized addiction); n[j] is n_std (standardized nicotine)
            a_h = clamp((1 - ψ) * a_h + ψ * n[j], A[1], A[end])
        end
    end

    return y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim
end


#############################
# MC-Specific Addiction
# Functions
#############################

"""
Estimate initial addiction stock for each household via fixed-point iteration.
MC version: takes hh_codes as a vector argument instead of reading CSV.

Returns:
- a0: Dict mapping household_code → estimated initial addiction stock
- max_iters: Maximum iterations to convergence across all households
"""
function get_initial_addiction_stock_mc(
    ψ::Real,
    A::AbstractVector{<:Real},
    n::AbstractVector{<:Real},
    y::AbstractVector{<:Integer},
    hh_codes::AbstractVector{<:Integer};
    max_iter::Integer = 500,
    tol::Real = 1e-6
)

    # Number of observations
    N = length(y)

    # Create a dictionary of vectors
    # Keys are household codes, values are observation indices
    hh_obs = Dict{Int, Vector{Int}}()
    for i in 1:N
        hh = hh_codes[i]
        if !haskey(hh_obs, hh)
            hh_obs[hh] = Int[]
        end
        push!(hh_obs[hh], i)
    end

    # Initialize a0 = 0 for all households
    a0 = Dict{Int, Float64}(hh => 0.0 for hh in keys(hh_obs))

    # Track iterations until convergence for each household
    hh_idx = 1
    iters_to_convergence = zeros(Int, length(hh_obs))
    n_not_converged = 0

    # Loop over households
    for (hh, obs_indices) in hh_obs

        # Iterate until convergence
        for iter in 1:max_iter

            # Simulate forward from current a0
            a = a0[hh]
            for i in obs_indices
                a = addiction_evolution(ψ, a, n[y[i]])
            end

            # Update a0 to terminal addiction level
            change = abs(a - a0[hh])
            a0[hh] = a

            # Check convergence
            if change < tol
                iters_to_convergence[hh_idx] = iter
                break
            end

            # Track non-convergence
            if iter == max_iter
                iters_to_convergence[hh_idx] = max_iter
                n_not_converged += 1
            end
        end

        hh_idx += 1
    end

    # Print single summary if any households did not converge
    if n_not_converged > 0
        log_msg("WARNING: Initial addiction fixed-point did not converge for $n_not_converged / $(length(hh_obs)) households (max_iter=$max_iter, tol=$tol)")
    end

    return a0, maximum(iters_to_convergence)
end


"""
Simulate household addiction trajectories and map to nearest grid indices.
MC version: takes hh_codes as a vector argument instead of reading CSV.

Returns:
- a_state: Vector of addiction grid indices for each observation
- a_continuous: Vector of actual continuous addiction levels
"""
function simulate_addiction_trajectories_mc(
    N_A::Integer,
    ψ::Real,
    A::AbstractVector{<:Real},
    n::AbstractVector{<:Real},
    y::AbstractVector{<:Integer},
    a0::AbstractDict,
    hh_codes::AbstractVector{<:Integer}
)

    # Number of observations
    N = length(y)

    # Initialize output arrays
    a_state      = Vector{Int}(undef, N)
    a_continuous = Vector{Float64}(undef, N)

    # Track current addiction level per household
    a_current = Dict{Int, Float64}()

    # Loop over observations
    for i in 1:N

        hh = hh_codes[i]

        # Get current addiction level (initial stock for first observation)
        a = get(a_current, hh, a0[hh])

        # Store continuous addiction level
        a_continuous[i] = a

        # Map to nearest grid index
        a_state[i] = argmin(abs.(A .- a))

        # Evolve addiction
        a_prime = addiction_evolution(ψ, a, n[y[i]])
        a_prime = clamp(a_prime, A[1], A[end])

        # Store for next period
        a_current[hh] = a_prime
    end

    return a_state, a_continuous
end


#############################
# MC Objective Function
#############################

# Global variables set by the MC loop before each estimation
y_sim              = Int[]
tya_state_sim      = Int[]
p_continuous_sim   = zeros(Float64, 0, 2)
hh_codes_sim       = Int[]

# Global file handle for parameter trace (set by the calling simulation script)
param_trace_io = nothing

# Global replication number (updated by MC loop)
current_replication = 0

"""
Helper: determine whether to print verbose output for this evaluation.
Prints evals 1-10 inclusive, then every 50th eval. Always prints penalties.
"""
function should_print_eval(eval_num::Integer)
    return eval_num <= 10 || eval_num % 50 == 0
end

"""
Write a single row to the parameter trace file.
Format: sim, outer_try, inner_run, eval, NLL, VFI_iters, θ_1, ..., θ_K
inner_run is the inner run number (1, 2, ...) or "long" for the convergence run.
"""
function write_param_trace(eval_num, nll, vfi_iters, θ_vec)
    global param_trace_io
    if param_trace_io === nothing
        return
    end
    inner_str = ra_inner_run == 0 ? "long" : string(ra_inner_run)
    θ_str = join([@sprintf("%.10f", x) for x in θ_vec], ",")
    println(param_trace_io, "$current_replication,$ra_outer_try,$inner_str,$eval_num,$(@sprintf("%.6f", nll)),$vfi_iters,$θ_str")
    flush(param_trace_io)
end

"""
MC-specific objective function for the optimizer with β estimation.

Same as objective() in 01_Functions_Beta.jl but uses simulated data (global
variables y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim) and MC-specific
addiction functions that take hh_codes directly instead of reading CSV.

θ_vec contains 14 parameters: 13 structural + β (present bias) as the 14th.
ψ is fixed (from reduced-form AR(1) estimate) and accessed via the global
variable set in the calling script.
Π_tya (4×4 TYA transition matrix) is accessed as a global variable.

Writes every evaluation's parameter vector to the trace file. Only prints
verbose log output at eval 1, 10, and every 50th eval thereafter.
"""
function objective_mc(θ_vec::AbstractVector{<:Real})
    global eval_count

    t_eval = time()
    eval_count += 1

    # Economic parameter bounds: α_T,α_E,α_TE,μ ≥ 0; γ,ω ≤ 0; β ∈ [0.01, 1.00].
    # Return penalty without solving VFI to save time.
    in_bounds, violations = check_parameter_bounds(θ_vec, param_names)
    if !in_bounds
        nll = 1e14
        write_param_trace(eval_count, nll, 0, θ_vec)
        if should_print_eval(eval_count)
            log_msg(@sprintf("  Eval %d | PENALTY (bounds: %s) | time = %.1fs",
                eval_count, violations, time() - t_eval))
        end
        return nll
    end

    # Extract β from θ_vec (14th element)
    β_current = θ_vec[end]

    # Structural parameters for flow utility (first 13 elements)
    θ_struct = θ_vec[1:end-1]

    # ψ is fixed (global from get_fixed_parameters)
    N_A_current, A_current = get_addiction_space(ψ)

    # Compute flow utility for the current θ (13 structural elements)
    U_current = get_flow_utility(
        θ_struct, N_J, N_A_current, N_Pcomb, A_current, c_cig, c_ecig, c_bundle, n, is_non_fda_flavored, is_fda_flavored, cat_idx, E
    )

    # Compute addiction transition brackets for the fixed ψ and A
    a_lower_current, a_upper_current, a_weight_current = precompute_addiction_transitions(
        N_J, N_A_current, ψ, A_current, n
    )

    # Determine whether to print verbose output for this eval
    print_this = should_print_eval(eval_count)

    # Solve VFI with β and TYA transitions (each evaluation starts fresh from zeros)
    V, V_decision_current, vfi_iters, vfi_converged = solve_vfi_sophisticated(
        N_J, N_A_current, N_P, N_Pcomb, β_current, δ, U_current,
        a_lower_current, a_upper_current, a_weight_current,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w,
        Π_tya;
        verbose = print_this
    )

    # Early-exit: if VFI did not converge, skip LL and return penalty
    if !vfi_converged
        elapsed = time() - t_eval
        nll = 1e14

        # Always log penalties and write to trace
        θ_str = join([@sprintf("%.6f", x) for x in θ_vec], ", ")
        log_msg(@sprintf("  Eval %d | PENALTY (VFI not converged) | VFI iters = %d | time = %.1fs | θ = [%s]",
            eval_count, vfi_iters, elapsed, θ_str))
        write_param_trace(eval_count, nll, vfi_iters, θ_vec)
        return nll
    end

    # Compute addiction trajectories using MC-specific functions with fixed ψ
    a0_current, _ = get_initial_addiction_stock_mc(ψ, A_current, n, y_sim, hh_codes_sim)
    _, a_continuous_current = simulate_addiction_trajectories_mc(
        N_A_current, ψ, A_current, n, y_sim, a0_current, hh_codes_sim
    )

    # Compute log-likelihood via trilinear interpolation at continuous states
    LL = log_likelihood(
        V_decision_current, N_J, N_P, A_current, P, y_sim, tya_state_sim,
        a_continuous_current, p_continuous_sim
    )

    nll = -LL
    elapsed = time() - t_eval

    # Write every evaluation to the trace file
    write_param_trace(eval_count, nll, vfi_iters, θ_vec)

    # Only print verbose output at eval 1, 10, and every 50th
    if print_this
        θ_str = join([@sprintf("%.6f", x) for x in θ_vec], ", ")
        log_msg(@sprintf("  Eval %d | NLL = %.4f | VFI iters = %d | time = %.1fs | θ = [%s]",
            eval_count, nll, vfi_iters, elapsed, θ_str))
    end

    # Return negative log-likelihood (optimizer minimizes)
    return nll
end
