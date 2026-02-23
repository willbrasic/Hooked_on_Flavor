################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script creates functions specific to the Monte Carlo simulation:
# simulate_data() for generating choices from a known DGP, and
# objective_mc() for evaluating the MC-specific objective function.
#
# When ESTIMATE_BETA = true (set in the calling script before include()),
# objective_mc() extracts β from θ_vec[end] and passes the remaining
# structural parameters to get_flow_utility(). Otherwise, β is the global
# fixed value.
#
# When ESTIMATE_PSI = true (set in the calling script before include()),
# objective_mc() extracts ψ from θ_vec, recomputes addiction transitions
# and trajectories at the candidate ψ. Otherwise, ψ is the global fixed value
# and pre-computed addiction objects are used.
#
# Parameter vector ordering:
#   Base:      [13 structural]
#   PSI only:  [13 structural, ψ]
#   BETA only: [13 structural, β]
#   Both:      [13 structural, ψ, β]  (β always last when estimated)
################################################################################


#############################
# Logging
#############################

# Uses log_io and log_msg() from 01_Functions.jl (included before this file).


#############################
# Warm-Start State
#############################

# V_warm stores the converged value function from the previous VFI solve
# within a Nelder-Mead run for warm-starting. Reset to nothing at the start
# of each outer try (L), inner run (M), and long run.
V_warm = nothing

# Tracks the current (outer_try, inner_run) phase from random_amoeba.
# When either changes, V_warm is reset to nothing.
last_ra_phase = (0, 0)


#############################
# Data Simulation
#############################

"""
Simulate household choices from a known DGP using real observed data.

Design-based MC simulation: conditions on real observables (prices, TYA,
panel structure) and only simulates choices from the model. 

Two-pass approach (matches actual estimation):
  Pass 1: Simulate choices starting from a₀ = 0 to get preliminary choices
  Pass 2: Use preliminary choices to compute a₀ via fixed-point iteration,
           then re-simulate choices from the corrected a₀

The two-pass approach is necessary because the initial addiction stock a₀
is unobserved. In the actual estimation, a₀ is recovered from the data
via fixed-point iteration on the observed choice sequence. Here we face
the same problem: we need choices to compute a₀, but we need a₀ to
simulate choices. Pass 1 breaks the circularity by starting from zero,
generating a preliminary choice sequence that is "close enough" for the
fixed-point iteration to converge to the correct a₀. Pass 2 then
re-simulates from the corrected a₀, producing the final simulated data.

Arguments:
- V_decision_true:   True decision-utility values V_d[tya, j, a, p] from VFI at θ_true
- ψ:                 Addiction decay rate (fixed)
- N_J:               Number of alternatives (40)
- N_P:               Number of price grid points per category (10)
- A:                 Addiction grid vector (length N_A, normalized to [0, 1])
- P:                 Price grid matrix (N_P × 2, columns = cig, ecig)
- n:                 Standardized nicotine vector by alternative (length N_J)
- real_p_continuous:  Observed continuous prices from real data (N_obs × 2)
- real_tya_state:     Observed TYA states from real data (length N_obs)
- real_hh_codes:      Observed household codes from real data (length N_obs)

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

    # Total number of observations (household-months) in the real data
    N_obs = length(real_hh_codes)

    # Copy real observables into output arrays. The simulation only generates
    # new choices (y_sim); prices, TYA states, and household codes are taken
    # directly from the real data to preserve realistic cross-sectional variation.
    tya_state_sim    = copy(real_tya_state)
    p_continuous_sim = copy(real_p_continuous)
    hh_codes_sim     = copy(real_hh_codes)

    # Build a dictionary mapping each household code to its observation indices.
    # This groups the panel so we can simulate each household's choice sequence
    # chronologically, evolving addiction forward from a₀.
    hh_obs = Dict{Int, Vector{Int}}()
    for i in 1:N_obs

        # Extract household code for observation i
        hh = real_hh_codes[i]

        # Create a new entry if this household hasn't been seen yet
        if !haskey(hh_obs, hh)
            hh_obs[hh] = Int[]
        end

        # Append this observation's index to the household's list
        push!(hh_obs[hh], i)
    end

    # Sort household codes for deterministic iteration order. Julia's Dict
    # has no guaranteed iteration order, so sorting ensures reproducibility
    # across Julia versions given the same RNG seed.
    sorted_hh_keys = sort(collect(keys(hh_obs)))

    # Pass 1: Simulate choices starting from a₀ = 0
    # This generates a preliminary choice sequence used to estimate a₀
    # via fixed-point iteration. The starting value a₀ = 0 is arbitrary
    # but the fixed-point iteration converges regardless of the initial guess
    # (contraction rate (1-ψ)^T).
    y_prelim = Vector{Int}(undef, N_obs)
    for hh in sorted_hh_keys
        obs_indices = hh_obs[hh]

        # Initialize addiction at zero for this household
        a_h = 0.0

        # Simulate choices chronologically within this household
        for i in obs_indices

            # Trilinearly interpolate V_decision at this observation's continuous
            # state (tya, addiction, cig price, ecig price) for all N_J alternatives
            v_interp = interpolate_v_choice(
                V_decision_true, real_tya_state[i], a_h,
                real_p_continuous[i, 1], real_p_continuous[i, 2],
                N_J, N_P, A, P
            )

            # Compute logit choice probabilities via stable softmax:
            # subtract the max to prevent overflow, exponentiate, normalize
            v_max = maximum(v_interp)
            exp_v = exp.(v_interp .- v_max)
            probs = exp_v ./ sum(exp_v)

            # Draw a choice from the categorical distribution
            j = categorical_sample(probs)

            # Store the preliminary choice for this observation
            y_prelim[i] = j

            # Evolve addiction forward: ã' = (1-ψ)·ã + ψ·n[j], clamped to grid bounds.
            # a_h is ã (normalized addiction); n[j] is n_std (standardized nicotine).
            a_h = clamp((1 - ψ) * a_h + ψ * n[j], A[1], A[end])
        end
    end

    # Compute a₀ via fixed-point iteration using preliminary choices 
    # Starting from the preliminary choice sequence, iterate the addiction law of
    # motion until the terminal addiction level equals the initial level (steady state).
    # This recovers each household's ergodic initial addiction stock.
    a0, _ = get_initial_addiction_stock(ψ, A, n, y_prelim, real_hh_codes)

    # Pass 2: Re-simulate choices from corrected a₀ 
    # Now that we have the correct initial addiction stocks, re-simulate the
    # entire choice sequence. These are the final simulated choices that the
    # estimator will try to recover parameters from.
    y_sim = Vector{Int}(undef, N_obs)
    for hh in sorted_hh_keys
        obs_indices = hh_obs[hh]

        # Initialize addiction at the fixed-point a₀ for this household
        a_h = a0[hh]

        # Simulate choices chronologically within this household
        for i in obs_indices

            # Trilinearly interpolate V_decision at this observation's continuous state
            v_interp = interpolate_v_choice(
                V_decision_true, real_tya_state[i], a_h,
                real_p_continuous[i, 1], real_p_continuous[i, 2],
                N_J, N_P, A, P
            )

            # Compute logit choice probabilities via stable softmax
            v_max = maximum(v_interp)
            exp_v = exp.(v_interp .- v_max)
            probs = exp_v ./ sum(exp_v)

            # Draw a choice from the categorical distribution
            j = categorical_sample(probs)

            # Store the final simulated choice for this observation
            y_sim[i] = j

            # Evolve addiction forward: ã' = (1-ψ)·ã + ψ·n[j], clamped to grid bounds
            a_h = clamp((1 - ψ) * a_h + ψ * n[j], A[1], A[end])
        end
    end

    return y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim
end


#############################
# MC Objective Function
#############################

"""
Determine whether to print verbose output for this evaluation.
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

    # Access the global file handle for the parameter trace CSV
    global param_trace_io

    # If no trace file is open (e.g., during testing), skip writing
    if param_trace_io === nothing
        return
    end

    # ra_inner_run is a global set by random_amoeba() in 01_Functions.jl:
    # 0 = long convergence run, 1,2,... = short inner runs
    inner_str = ra_inner_run == 0 ? "long" : string(ra_inner_run)

    # Format each parameter to 10 decimal places, joined by commas
    θ_str = join([@sprintf("%.10f", x) for x in θ_vec], ",")

    # Write one CSV row: sim, outer_try, inner_run, eval, NLL, VFI_iters, θ_1, ..., θ_D 
    # current_replication and ra_outer_try are globals set by the MC loop and random_amoeba()
    println(param_trace_io, "$current_replication,$ra_outer_try,$inner_str,$eval_num,$(@sprintf("%.6f", nll)),$vfi_iters,$θ_str")

    # Flush immediately so partial results survive if the job is killed
    flush(param_trace_io)
end

"""
MC-specific objective function for the optimizer.

Same as objective() in 01_Functions.jl but uses simulated data (global variables
y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim) instead of CSV data.

Parameter vector ordering: [13 structural, ψ (if ESTIMATE_PSI), β (if ESTIMATE_BETA)]

Writes every evaluation's parameter vector to the trace file. Only prints
verbose log output at eval 1, 10, and every 50th eval thereafter.

Pre-computed addiction globals (set by 02_MC_Simulation_Array.jl, used when ESTIMATE_PSI = false):
  N_A, A                                             — addiction grid (existing globals)
  mc_a_lower_fixed, mc_a_upper_fixed, mc_a_weight_fixed — addiction transition brackets
  mc_a_continuous_fixed                              — continuous addiction trajectories
When ESTIMATE_PSI = true, these are recomputed at the candidate ψ each evaluation.
"""
function objective_mc(θ_vec::AbstractVector{<:Real})

    # Use global evaluation counter
    global eval_count

    # Start evaluation time
    t_eval = time()

    # Update evaluation count
    eval_count += 1

    # Economic parameter bounds check
    # If any parameter falls outside their respective bounds, return penalty (very large number for LL)
    in_bounds, violations = check_parameter_bounds(θ_vec, param_names)
    if !in_bounds

        # Large LL
        nll = 1e14

        # Write output
        write_param_trace(eval_count, nll, 0, θ_vec)

        # Print and log message
        if should_print_eval(eval_count)

            log_msg(@sprintf("  Eval %d | PENALTY (bounds: %s) | time = %.1fs", eval_count, violations, time() - t_eval))

        end

        return nll
    end

    # Extract β and ψ from θ_vec based on which flags are active.
    # Parameter ordering: [13 structural, ψ (if ESTIMATE_PSI), β (if ESTIMATE_BETA)]
    if ESTIMATE_BETA && ESTIMATE_PSI
        β_current = θ_vec[end]
        ψ_current = θ_vec[end-1]
        θ_struct = θ_vec[1:end-2]
    elseif ESTIMATE_BETA
        β_current = θ_vec[end]
        ψ_current = ψ
        θ_struct = θ_vec[1:end-1]
    elseif ESTIMATE_PSI
        ψ_current = θ_vec[end]
        β_current = β
        θ_struct = θ_vec[1:end-1]
    else
        β_current = β
        ψ_current = ψ
        θ_struct = θ_vec
    end

    # When ESTIMATE_PSI = true, recompute addiction objects at the candidate ψ.
    # These shadow the pre-computed globals (mc_a_lower_fixed, etc.) from 02_MC_Simulation_Array.jl.
    if ESTIMATE_PSI
        N_A_cur, A_cur = get_addiction_space(ψ_current)
        a_lower_cur, a_upper_cur, a_weight_cur = precompute_addiction_transitions(N_J, N_A_cur, ψ_current, A_cur, n)
        a0_cur, _ = get_initial_addiction_stock(ψ_current, A_cur, n, y_sim, hh_codes_sim)
        _, a_continuous_cur = simulate_addiction_trajectories(N_A_cur, ψ_current, A_cur, n, y_sim, hh_codes_sim, a0_cur)
    else
        N_A_cur = N_A
        A_cur = A
        a_lower_cur = mc_a_lower_fixed
        a_upper_cur = mc_a_upper_fixed
        a_weight_cur = mc_a_weight_fixed
        a_continuous_cur = mc_a_continuous_fixed
    end

    # Compute flow utility for the current θ (structural parameters only, excludes β and ψ)
    U_current = get_flow_utility(
        θ_struct, N_J, N_A_cur, N_Pcomb, A_cur, q_cig, q_ecig, q_bundle, n, is_flavored, is_fda_flavored, cat_idx, E
    )

    # Determine whether to print verbose output for this eval
    print_this = should_print_eval(eval_count)

    # Warm-start: reuse the previous V as initial guess within a NM run.
    # Reset V_warm when the optimizer phase changes (new outer try, inner run, or long run).
    if WARM_START

        # Access global value function
        global V_warm, last_ra_phase

        # Get current outer and inner try
        current_phase = (ra_outer_try, ra_inner_run)

        # If the outer and inner runs are different than before
        if current_phase != last_ra_phase

            # Reset the value function
            V_warm = nothing

            # Update the outer and inner run
            last_ra_phase = current_phase
        end

        # Update the value function
        V_init_current = V_warm
    else

        # If not doing warm start, then set to nothing
        V_init_current = nothing
    end

    # VFI
    # When ESTIMATE_PSI = false, a_lower_cur etc. are the pre-computed globals.
    # When ESTIMATE_PSI = true, they are recomputed above at the candidate ψ.
    V, V_decision_current, vfi_iters, vfi_converged = solve_vfi_sophisticated(
        N_J, N_A_cur, N_P, N_Pcomb, β_current, δ, U_current,
        a_lower_cur, a_upper_cur, a_weight_cur,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w,
        Π;
        V_init = V_init_current,
        verbose = print_this
    )

    # If VFI did not converge, skip LL and return penalty
    if !vfi_converged

        # Get elapsed time
        elapsed = time() - t_eval
        nll = 1e14

        # Print and log message
        θ_str = join([@sprintf("%.6f", x) for x in θ_vec], ", ")
        log_msg(@sprintf("  Eval %d | PENALTY (VFI not converged) | VFI iters = %d | time = %.1fs | θ = [%s]",
            eval_count, vfi_iters, elapsed, θ_str))

        # Write output to CSV
        write_param_trace(eval_count, nll, vfi_iters, θ_vec)

        return nll
    end

    # Store converged V for warm-starting the next evaluation within this NM run.
    # Only store when VFI converged — unconverged V (penalty case) is not stored.
    if WARM_START
        V_warm = V
    end

    # Compute log-likelihood via trilinear interpolation at continuous states
    LL = log_likelihood(
        V_decision_current, N_J, N_P, A_cur, P, y_sim, tya_state_sim,
        a_continuous_cur, p_continuous_sim
    )

    nll = -LL

    # Get elapsed time
    elapsed = time() - t_eval

    # Write every evaluation to the trace file
    write_param_trace(eval_count, nll, vfi_iters, θ_vec)

    # Print and log message for current objective evaluation
    if print_this
        θ_str = join([@sprintf("%.6f", x) for x in θ_vec], ", ")
        log_msg(@sprintf("  Eval %d | NLL = %.4f | VFI iters = %d | time = %.1fs | θ = [%s]",
            eval_count, nll, vfi_iters, elapsed, θ_str))
    end

    # Return negative log-likelihood (optimizer minimizes)
    return nll
end
