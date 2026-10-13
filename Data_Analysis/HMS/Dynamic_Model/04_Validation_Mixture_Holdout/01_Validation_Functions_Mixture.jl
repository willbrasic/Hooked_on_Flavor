################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# Instead of using predicted choice probabilities directly, this approach
# simulates forward choice sequences from the model: at each month, draw a
# choice from the logit probabilities, then evolve the addiction stock based
# on the simulated choice (not the observed choice). This produces a stricter
# test of model fit because the model must generate realistic dynamic paths.
#
# Observed prices and TYA states are used (not simulated). Only the addiction
# stocks evolve endogenously based on simulated choices.
################################################################################


################################################################################
# Table of Contents
#
#  1. Period Indices
#  2. Predicted Choice Probabilities
#  3. Core Simulation Engine
#  4. Streak Persistence (Simulated Data)
#  5. Run S Simulations and Average
#  6. Post-Purchase Path Event Study
#  7. Run Post-Purchase Path Validation
################################################################################


#############################
# 1. Period Indices
#############################

"""
Extract purchase months from Teen_Young_Adult.csv and convert to integer
period indices (1, 2, 3, ...) for time-series aggregation.

Integer indices are required because the streak persistence and post-purchase
path functions detect consecutive months via period_idx[i+1] == period_idx[i] + 1.

Returns:
- period_idx:    Vector of period indices for each observation
- period_labels: Vector of calendar month labels (e.g., "2021-01")
- N_periods:     Number of unique periods
"""
function get_period_indices(;
    file_name::AbstractString = "./TYA_States.csv"
)

    df = CSV.read(file_name, DataFrame)

    # Extract purchase month strings
    months_raw = string.(df.purchase_month)

    # Get unique months in sorted order
    unique_months = sort(unique(months_raw))
    N_periods = length(unique_months)

    # Create month → index mapping
    month_to_idx = Dict{String, Int}()
    for (idx, m) in enumerate(unique_months)
        month_to_idx[m] = idx
    end

    # Map each observation to its period index
    period_idx = [month_to_idx[m] for m in months_raw]

    # Clean labels
    period_labels = [m[1:7] for m in unique_months]

    return period_idx, period_labels, N_periods
end


#############################
# 2. Predicted Choice
# Probabilities
#############################

"""
Compute predicted choice probabilities at all observed states under a given
V_choice solution.

This function is used where probabilities must be evaluated at OBSERVED (not
simulated) addiction states. The posterior type weight computation requires
evaluating each type's likelihood at the observed states to apply Bayes' rule,
and the in-sample price elasticity exercise (04_Validation_Mixture only) needs
aggregate choice probabilities at the observed states under the baseline and
price-shocked V_choice solutions. The forward simulation does NOT use this
function; it calls interpolate_v_choice directly at the evolving SIMULATED
(not observed) addiction state.

For each observation i at their current state:
  1. Interpolate V_choice at the continuous state for all N_J alternatives
  2. Compute softmax choice probabilities

Returns:
- probs: Matrix{Float64} of size (N_obs × N_J), choice probabilities
"""
function compute_predicted_probs(
    V_choice::Array{Float64, 6},
    tya_state::AbstractVector{<:Integer},
    af_continuous::AbstractVector{<:Real},
    as_continuous::AbstractVector{<:Real},
    aflav_continuous::AbstractVector{<:Real},
    p_continuous::AbstractMatrix{<:Real},
    N_J::Integer,
    N_P::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real}
)

    N_obs = length(tya_state)

    # Initialize output
    probs = Matrix{Float64}(undef, N_obs, N_J)

    for i in 1:N_obs

        # Interpolate V_choice at continuous state
        v_interp = interpolate_v_choice(
            V_choice, tya_state[i], af_continuous[i], as_continuous[i],
            aflav_continuous[i], p_continuous[i, 1], p_continuous[i, 2],
            N_J, N_P, A_f, A_s, A_flav, P
        )

        # Fix NaN from 0.0 * (-Inf)
        # Banned alternatives have V_choice = -Inf at all grid points;
        # when an interpolation weight is exactly 0, 0.0 * (-Inf) = NaN.
        # Replace with -Inf so exp(-Inf) = 0 in subsequent softmax/logsumexp.
        replace!(v_interp, NaN => -Inf)

        # Softmax choice probabilities.
        # I shift by v_max before exponentiating to prevent numerical overflow.
        # The ratio exp(v_j - v_max) / sum(exp(v_k - v_max)) is mathematically
        # identical to exp(v_j) / sum(exp(v_k)), but when v_j is very large (e.g.,
        # 700+), exp(v_j) = Inf. Subtracting v_max makes every argument
        # <= 0, so exp() stays finite.
        v_max = maximum(v_interp)
        v_shifted = v_interp .- v_max
        exp_v = exp.(v_shifted)
        sum_exp_v = sum(exp_v)
        for j in 1:N_J
            probs[i, j] = exp_v[j] / sum_exp_v
        end
    end

    return probs
end


#############################
# 3. Core Simulation
#############################

"""
Simulate choice sequences for all households.

The key distinction from `compute_predicted_probs` is that it evaluates
V_choice at the evolving SIMULATED addiction states rather than the observed
state: the type draw is stochastic, prices and TYA are held at observed values,
but addiction stocks grow endogenously from the sequence of simulated choices.
Validating against the observed-state probabilities from `compute_predicted_probs`
would only confirm in-sample fit at observed states. The stricter test is
whether the model's own dynamics are internally consistent. In the forward
simulation, the addiction stock at month t is determined by what the model
itself chose in months 1 through t-1, not by what the household actually chose.
If the model slightly overestimates purchase rates, stocks compound upward;
elevated stocks then reduce withdrawal disutility (gamma_1 × a at the outside
option), which further raises purchase probabilities in the next period. A
misspecified addiction process amplifies this feedback over the horizon. The
streak persistence curves test exactly this property: if the model correctly
captures both the flow utility and the law of motion, simulated repurchase rates
conditional on streak length should match the empirical pattern that longer
streaks predict higher continuation, without being given the observed stocks to
lean on.

For each household:
  1. Assign type (1, 2, or 3) by drawing from posterior type probabilities
  2. Initialize addiction at the observed initial stock (af0, as0, aflav0)
  3. For each month t:
     a. Interpolate V_choice at the SIMULATED addiction state + observed (tya, prices)
     b. Compute softmax choice probabilities
     c. Draw a choice via categorical sampling
     d. Evolve addiction based on the simulated choice

Returns:
- sim_y: Vector{Int} of simulated choices (same length as observed data)


"""
function simulate_household_sequences_mixture(
    V_choice_1::Array{Float64, 6},
    V_choice_2::Array{Float64, 6},
    V_choice_3::Array{Float64, 6},
    hh_posterior::AbstractMatrix{<:Real},
    hh_ranges::AbstractVector{<:Tuple{Int, Int}},
    tya_state::AbstractVector{<:Integer},
    p_continuous::AbstractMatrix{<:Real},
    hh_codes::AbstractVector{<:Integer},
    af0::AbstractDict,
    as0::AbstractDict,
    aflav0::AbstractDict,
    n::AbstractVector{<:Real},
    n_flav::AbstractVector{<:Real},
    ψ_2::Real,
    ψ_1::Real,
    ψ_3::Real,
    N_J::Integer,
    N_P::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    rng::AbstractRNG
)

    N_obs = length(tya_state)
    N_HH = length(hh_ranges)

    # Output: simulated choice for each observation
    sim_y = Vector{Int}(undef, N_obs)

    # Pre-allocate interpolation buffer
    v_interp = Vector{Float64}(undef, N_J)

    for h in 1:N_HH

        start_idx, stop_idx = hh_ranges[h]

        # hh_posterior[h, k] is the posterior probability that household h
        # belongs to type k, computed via Bayes' rule:
        #   P(k | y_h) proportional to π_k(tya_share_h) × L_k(y_h | θ_k)
        # where π_k is the mixture weight from the softmax and L_k is the
        # type-k likelihood evaluated at the observed choice sequence y_h.
        # Each row sums to 1 across K=3 columns.
        #
        # I assign a type for this draw by sampling from the posterior rather
        # than taking the argmax (most likely type). The difference matters for
        # households whose posteriors are not concentrated: a household with
        # posterior (0.45, 0.35, 0.20) would always be assigned to type 1 under
        # argmax, ignoring the 55% probability that it belongs to another type.
        # Stochastic assignment instead draws type 1 with probability 0.45,
        # type 2 with 0.35, and type 3 with 0.20. So, across S draws the
        # household's simulated sequences average over its type uncertainty.
        #
        # Implementation: draw u ~ U[0,1], then select the type whose
        # cumulative posterior interval contains u.
        u = rand(rng)
        p1 = hh_posterior[h, 1]
        p2 = hh_posterior[h, 2]
        V_choice = if u < p1
            V_choice_1
        elseif u < p1 + p2
            V_choice_2
        else
            V_choice_3
        end

        # Initialize addiction at observed initial stocks
        hh_code = hh_codes[start_idx]
        a_f_h = af0[hh_code]
        a_s_h = as0[hh_code]
        a_flav_h = aflav0[hh_code]

        for i in start_idx:stop_idx

            # Interpolate V_choice at the SIMULATED addiction state (not observed).
            # The addiction stock evolves from simulated choices, so V_choice is evaluated
            # at the state the household would reach under the model.
            v_interp .= interpolate_v_choice(
                V_choice, tya_state[i], a_f_h, a_s_h, a_flav_h,
                p_continuous[i, 1], p_continuous[i, 2],
                N_J, N_P, A_f, A_s, A_flav, P
            )

            # Fix NaN from 0.0 * (-Inf)
            replace!(v_interp, NaN => -Inf)

            # Softmax choice probabilities (shift by v_max to prevent overflow).
            v_max = maximum(v_interp)
            exp_v = exp.(v_interp .- v_max)
            sum_exp_v = sum(exp_v)
            exp_v ./= sum_exp_v

            # Draw a choice via the inverse CDF method.
            # u_draw is uniform on [0,1]; I walk through cumulative probabilities
            # until I exceed u_draw.
            u_draw = rand(rng)
            cumulative = 0.0
            j_sim = N_J  # fallback if rounding prevents the last alternative from triggering
            for j in 1:N_J
                cumulative += exp_v[j]
                if u_draw <= cumulative
                    j_sim = j
                    break
                end
            end
            sim_y[i] = j_sim

            # Evolve all three addiction stocks based on the simulated choice.
            # Law of motion: a' = (1 - ψ) * a + ψ * n_j, where n_j is the nicotine
            # (or flavor) content of alternative j. Stocks are clamped to [A[1], A[end]]
            # after each step because the value function grid only covers that range,
            # and interpolating outside it would silently extrapolate.
            a_f_h = addiction_evolution(ψ_2, a_f_h, n[j_sim])
            a_f_h = clamp(a_f_h, A_f[1], A_f[end])      # fast stock in [0, 1]

            a_s_h = addiction_evolution(ψ_1, a_s_h, n[j_sim])
            a_s_h = clamp(a_s_h, A_s[1], A_s[end])      # slow stock in [0, 1]

            a_flav_h = addiction_evolution(ψ_3, a_flav_h, n_flav[j_sim])
            a_flav_h = clamp(a_flav_h, A_flav[1], A_flav[end])  # flavored habit stock in [0, 1]
        end
    end

    return sim_y
end


#############################
# 4. Streak Persistence from
# Simulated Data
#############################

"""
Compute streak-length continuation rates from simulated choice sequences,
for cigarettes and e-cigarettes only.

A streak is consecutive months of buying the same product. If a household
buys cigarettes in January, February, and March, their streak length in
March is 3.

A continuation rate is defined as of all the times any household had been buying
for exactly k months in a row, what fraction of them bought again the next
month? Example: if there are 100 observations where streak = 3 and 80 of
them bought again the next month, the continuation rate at streak-3 = 0.80.
This produces a single number between 0 and 1 for each streak length 1 to 12.


Example: household buys cigarettes in months 1 to 5; months 1 to 3 non-TYA,
months 4 to 5 TYA.

  Month | Streak | TYA? | Counted in TYA rates?
  ------|--------|------|---------------------------------------
    1   |   1    |  No  | No
    2   |   2    |  No  | No
    3   |   3    |  No  | No
    4   |   4    |  Yes | Yes - goes into streak-4 bucket
    5   |   5    |  Yes | Yes - but no month 6 to check, skipped

This household contributes one data point to the TYA continuation rate:
streak-4 continued. If instead streaks were reset at TYA status changes,
the TYA sequence would only contain months 4 and 5 with streak lengths 1
and 2, so month 4 would incorrectly enter the streak-1 bucket.

Returns Dictionary with keys "cig", "ecig", "flav_ecig", "orig_ecig" corresponding
to streak lengths for each of these categories.
"""
function compute_sim_streak_continuation(
    sim_y::AbstractVector{<:Integer},
    cat_idx::AbstractVector{<:Integer},
    hh_codes::AbstractVector,
    period_idx::AbstractVector{<:Integer};
    obs_mask::Union{AbstractVector{Bool}, Nothing} = nothing,
    max_streak::Integer = 12
)

    N_obs = length(sim_y)
    N_J = maximum(sim_y)

    # Define which alternatives count as "purchasing" for each product type
    cig_alts      = [cat_idx[j] in (1, 5, 6, 7) for j in 1:N_J]
    ecig_alts     = [cat_idx[j] in (2, 3, 4, 5, 6, 7) for j in 1:N_J]
    flav_ecig_alts = [cat_idx[j] in (3, 4, 6, 7) for j in 1:N_J]
    orig_ecig_alts = [cat_idx[j] in (2, 5) for j in 1:N_J]

    # Determine whether each simulated observation is a "purchase" month
    is_cig      = [cig_alts[sim_y[i]] for i in 1:N_obs]
    is_ecig     = [ecig_alts[sim_y[i]] for i in 1:N_obs]
    is_flav_ecig = [flav_ecig_alts[sim_y[i]] for i in 1:N_obs]
    is_orig_ecig = [orig_ecig_alts[sim_y[i]] for i in 1:N_obs]

    results = Dict{String, Matrix{Float64}}()

    for (label, is_purchase) in [("cig", is_cig), ("ecig", is_ecig), ("flav_ecig", is_flav_ecig), ("orig_ecig", is_orig_ecig)]

        # Compute streak lengths on the FULL data
        #
        # Streak logic per observation i:
        #   - Not a purchase month: streak resets to 0
        #   - First obs overall, or start of a new household, or a month(s) gap in the panel:
        #     start a new streak at 1
        #   - Consecutive purchase following another purchase in the same household: increment
        #   - Purchase after a non-purchase in the same continuous sequence: restart at 1
        streak = zeros(Int, N_obs)
        for i in 1:N_obs
            if !is_purchase[i]
                streak[i] = 0  # not a purchase month - no active streak
            elseif i == 1 || hh_codes[i] != hh_codes[i-1] || period_idx[i] != period_idx[i-1] + 1
                streak[i] = 1  # start of a new streak (first obs, new household, or gap)
            else
                streak[i] = is_purchase[i-1] ? streak[i-1] + 1 : 1  # extend or restart
            end
        end

        # Accumulate continuation counts by streak length.
        counts     = zeros(Float64, max_streak)
        continues  = zeros(Float64, max_streak)

        for i in 1:(N_obs - 1)

            k = streak[i]
            if k < 1
                continue  # not an active streak - nothing to count
            end

            # Only count if the next period is the same household and a consecutive month.
            # If the household drops out or there is a gap, I cannot observe whether
            # the streak continued, so I skip this event.
            if hh_codes[i] != hh_codes[i+1] || period_idx[i+1] != period_idx[i] + 1
                continue
            end

            # Apply observation mask: only count events at observations that pass the filter (E.g., TYA-present HHs)
            if obs_mask !== nothing && !obs_mask[i]
                continue
            end

            # Bin streak lengths at max_streak (long streaks are grouped into the last cell)
            k_bin = min(k, max_streak)
            counts[k_bin] += 1.0
            # Record whether the household continued purchasing in the next month
            continues[k_bin] += is_purchase[i+1] ? 1.0 : 0.0
        end

        # Build results matrix
        res = Matrix{Float64}(undef, max_streak, 3)
        for k in 1:max_streak
            res[k, 1] = Float64(k)
            res[k, 2] = counts[k] > 0 ? continues[k] / counts[k] : NaN
            res[k, 3] = counts[k]
        end

        results[label] = res
    end

    return results
end


#############################
# 5. Run S Simulations and
# Average Results
#############################

"""
Run S simulation draws and accumulate streak persistence statistics,
separately for all households and TYA households. The reported continuation
rate at each streak length is (total continuations across all S draws) /
(total qualifying events across all S draws). For example, if across 100
draws there were 3,000 total streak-5 events and 1,800 of them continued,
the rate is 1,800/3,000 = 0.60. The confidence band captures simulation
uncertainty: each of the S draws produces its own continuation rate at each
streak length. The 2.5th and 97.5th percentiles
of those S rates form the lower and upper bounds of the band, giving a 95%
confidence band.

Returns a NamedTuple with:
- streak_all:       Dict of matrices (cig, ecig, flav_ecig) for all households
                    Each matrix: max_streak × 3 [streak_length, mean_rate, avg_N]
- streak_tya:       Dict of matrices (cig, ecig, flav_ecig) for TYA households
- streak_all_ci:    Dict of matrices (cig, ecig, flav_ecig) for all households
                    Each matrix: max_streak × 2 [p05, p95]
- streak_tya_ci:    Dict of matrices (cig, ecig, flav_ecig) for TYA households
- streak_extra:     Dict{label => Dict of matrices} for each extra subgroup mask
                    passed via `extra_masks` (e.g. modal latent type). Same
                    layout as streak_all. Empty when no extra masks are given.
- streak_extra_ci:  Dict{label => Dict of CI matrices} for each extra subgroup

`extra_masks` is a vector of `label => obs_mask` pairs. Each mask is an
observation-level Bool vector; the simulation itself is unchanged (each draw
still samples every household's type from its posterior), only the
accumulation step is restricted to the masked observations.
"""
function run_simulation_validation(
    V_choice_1::Array{Float64, 6},
    V_choice_2::Array{Float64, 6},
    V_choice_3::Array{Float64, 6},
    hh_posterior::AbstractMatrix{<:Real},
    hh_ranges::AbstractVector{<:Tuple{Int, Int}},
    tya_state::AbstractVector{<:Integer},
    p_continuous::AbstractMatrix{<:Real},
    hh_codes::AbstractVector{<:Integer},
    af0::AbstractDict,
    as0::AbstractDict,
    aflav0::AbstractDict,
    n::AbstractVector{<:Real},
    n_flav::AbstractVector{<:Real},
    ψ_2::Real,
    ψ_1::Real,
    ψ_3::Real,
    N_J::Integer,
    N_P::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    cat_idx::AbstractVector{<:Integer},
    period_idx::AbstractVector{<:Integer},
    S::Integer;
    base_seed::Integer = 12345,
    max_streak::Integer = 12,
    extra_masks::AbstractVector{<:Pair} = Pair{String, Vector{Bool}}[]
)

    # Pre-compute TYA observation mask (binary: 1 = no TYA, 2 = TYA present)
    mask_tya = [tya_state[i] == 2 for i in eachindex(tya_state)]

    # Streak labels
    streak_labels = ["cig", "ecig", "flav_ecig", "orig_ecig"]

    # Accumulators for streak persistence (raw counts), one set per subgroup
    acc_counts_all       = Dict(l => zeros(Float64, max_streak) for l in streak_labels)
    acc_continues_all    = Dict(l => zeros(Float64, max_streak) for l in streak_labels)
    acc_counts_tya       = Dict(l => zeros(Float64, max_streak) for l in streak_labels)
    acc_continues_tya    = Dict(l => zeros(Float64, max_streak) for l in streak_labels)

    # Per-draw continuation rates for confidence bands (max_streak × S)
    draw_rates_all    = Dict(l => fill(NaN, max_streak, S) for l in streak_labels)
    draw_rates_tya    = Dict(l => fill(NaN, max_streak, S) for l in streak_labels)

    # Same accumulators for each extra subgroup (e.g. modal latent type), keyed by label
    extra_labels        = [String(first(pm)) for pm in extra_masks]
    acc_counts_extra    = Dict(el => Dict(l => zeros(Float64, max_streak) for l in streak_labels) for el in extra_labels)
    acc_continues_extra = Dict(el => Dict(l => zeros(Float64, max_streak) for l in streak_labels) for el in extra_labels)
    draw_rates_extra    = Dict(el => Dict(l => fill(NaN, max_streak, S) for l in streak_labels) for el in extra_labels)

    for s in 1:S

        if s % 10 == 0 || s == 1
            log_msg("  Simulation $s / $S...")
        end

        # Create RNG for this draw
        rng = MersenneTwister(base_seed + s)

        # Simulate choice sequences
        sim_y = simulate_household_sequences_mixture(
            V_choice_1, V_choice_2, V_choice_3, hh_posterior, hh_ranges,
            tya_state, p_continuous, hh_codes, af0, as0, aflav0,
            n, n_flav, ψ_2, ψ_1, ψ_3, N_J, N_P, A_f, A_s, A_flav, P, rng
        )

        # Compute streak continuation rates for all households (no subgroup filter).
        # compute_sim_streak_continuation returns, for each streak length k:
        #   col 1 = k, col 2 = continuation rate in this draw, col 3 = number of events
        # I do NOT average the per-draw rates directly. Instead I accumulate the raw
        # event counts (n_k) and raw continuation counts (rate * n_k) across all S draws,
        # then divide once at the end. This gives: final rate = total_continuations /
        # total_events across all S draws. Draws with more qualifying events at streak k
        # therefore contribute more to the final estimate than low-count draws.
        streak_all = compute_sim_streak_continuation(
            sim_y, cat_idx, hh_codes, period_idx;
            max_streak=max_streak
        )
        for l in streak_labels
            res = streak_all[l]
            for k in 1:max_streak
                n_k = res[k, 3]
                acc_counts_all[l][k]    += n_k
                acc_continues_all[l][k] += isnan(res[k, 2]) ? 0.0 : res[k, 2] * n_k
                draw_rates_all[l][k, s]  = res[k, 2]  # NaN if no events in this draw
            end
        end

        # Same accumulation for TYA-only months (obs_mask filters to tya_state == 2).
        # Streak lengths still reflect the full purchase history; only the accumulation
        # step is restricted to months where TYA was present.
        streak_tya = compute_sim_streak_continuation(
            sim_y, cat_idx, hh_codes, period_idx;
            obs_mask=mask_tya, max_streak=max_streak
        )
        for l in streak_labels
            res = streak_tya[l]
            for k in 1:max_streak
                n_k = res[k, 3]
                acc_counts_tya[l][k]    += n_k
                acc_continues_tya[l][k] += isnan(res[k, 2]) ? 0.0 : res[k, 2] * n_k
                draw_rates_tya[l][k, s]  = res[k, 2]
            end
        end

        # Same accumulation for each extra subgroup mask (e.g. modal type 1 / type 2
        # households). The mask only restricts which observations are counted.
        for (el, mask_extra) in extra_masks
            streak_extra_s = compute_sim_streak_continuation(
                sim_y, cat_idx, hh_codes, period_idx;
                obs_mask=mask_extra, max_streak=max_streak
            )
            for l in streak_labels
                res = streak_extra_s[l]
                for k in 1:max_streak
                    n_k = res[k, 3]
                    acc_counts_extra[el][l][k]    += n_k
                    acc_continues_extra[el][l][k] += isnan(res[k, 2]) ? 0.0 : res[k, 2] * n_k
                    draw_rates_extra[el][l][k, s]  = res[k, 2]
                end
            end
        end
    end

    # Compute the final point estimate: total continuations / total events across
    # all S draws.
    # Column 3 of the output matrix reports the average number of events per draw
    # (total count / S), which indicates how many observations supported each estimate.
    function compute_avg(acc_counts, acc_continues)
        avg = Dict{String, Matrix{Float64}}()
        for l in streak_labels
            res = Matrix{Float64}(undef, max_streak, 3)
            for k in 1:max_streak
                res[k, 1] = Float64(k)
                res[k, 2] = acc_counts[l][k] > 0 ? acc_continues[l][k] / acc_counts[l][k] : NaN
                res[k, 3] = acc_counts[l][k] / S
            end
            avg[l] = res
        end
        return avg
    end

    avg_streak_all    = compute_avg(acc_counts_all,    acc_continues_all)
    avg_streak_tya    = compute_avg(acc_counts_tya,    acc_continues_tya)
    avg_streak_extra  = Dict(el => compute_avg(acc_counts_extra[el], acc_continues_extra[el]) for el in extra_labels)

    # Compute confidence bands from the S per-draw continuation rates stored in
    # draw_rates_*.
    function compute_ci(draw_rates::Dict{String, Matrix{Float64}})
        ci = Dict{String, Matrix{Float64}}()
        for l in streak_labels
            res = Matrix{Float64}(undef, max_streak, 2)
            for k in 1:max_streak
                rates_k = filter(!isnan, draw_rates[l][k, :])
                if length(rates_k) >= 2
                    sort!(rates_k)
                    n_draws = length(rates_k)
                    idx_05 = max(1, Int(ceil(0.025 * n_draws)))
                    idx_95 = min(n_draws, Int(ceil(0.975 * n_draws)))
                    res[k, 1] = rates_k[idx_05]
                    res[k, 2] = rates_k[idx_95]
                else
                    res[k, 1] = NaN
                    res[k, 2] = NaN
                end
            end
            ci[l] = res
        end
        return ci
    end

    ci_all    = compute_ci(draw_rates_all)
    ci_tya    = compute_ci(draw_rates_tya)
    ci_extra  = Dict(el => compute_ci(draw_rates_extra[el]) for el in extra_labels)

    return (
        streak_all       = avg_streak_all,
        streak_tya       = avg_streak_tya,
        streak_all_ci    = ci_all,
        streak_tya_ci    = ci_tya,
        streak_extra     = avg_streak_extra,
        streak_extra_ci  = ci_extra
    )
end
