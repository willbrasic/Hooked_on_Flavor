################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# September 2026
#
# Functions for the Vuong (1989) non-nested model comparison between
# the beta = 1 (time-consistent) and beta < 1 (present-biased) specifications
# of the K=3 mixture dynamic discrete choice model.
#
# Both specifications are estimated with beta FIXED at its respective value
# rather than estimated, so neither model nests the other: they are two
# distinct parameterizations of the same likelihood evaluated on the same
# data. The Vuong test is the appropriate comparison in that setting.
#
# These functions depend on objects defined in
# 02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl, which must be
# included before this file.
################################################################################


#############################
# 1. Preliminaries
#############################

# Install any missing packages, then load
import Pkg
for pkg in ["Distributions"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end
using Distributions


#-------------------------------------------------------------------------------
# resolve_estimates_path
#-------------------------------------------------------------------------------
"""
    resolve_estimates_path(est_root, psi_tag, beta_tag)

Return the path to the saved point-estimate CSV for one beta value.

Estimation results are laid out differently on the HPC than they are locally,
so several candidate paths are tried in order and the first that exists is
returned. This mirrors how `05_CF_Mixture/02_CF_Mixture.jl` locates the same
files, with additional fallbacks for the local results tree.

An error listing every candidate is thrown when none exist, so a path problem
surfaces immediately rather than partway through a multi-hour job.
"""
function resolve_estimates_path(
    est_root::AbstractString,
    psi_tag::AbstractString,
    beta_tag::AbstractString
)

    # Base file name used by the estimation script
    stem    = "Dynamic_Model_Mixture_$(psi_tag)_$(beta_tag)"

    # Candidate locations, most specific first
    candidates = String[
        # HPC / CF-script layout: <root>/<stem>_Results/<stem>_Estimates/<stem>_Estimates.csv
        joinpath(est_root, "$(stem)_Results",    "$(stem)_Estimates",    "$(stem)_Estimates.csv"),

        # Flattened results directory
        joinpath(est_root, "$(stem)_Results",    "$(stem)_Estimates.csv"),

        # Local archive layout: <root>/02_Second_Stage_Estimation_Mixture_Results/<beta_tag>/<stem>_Estimates.csv
        joinpath(est_root, "02_Second_Stage_Estimation_Mixture_Results", beta_tag, "$(stem)_Estimates.csv"),

        # Beta-tagged subdirectory directly under the root
        joinpath(est_root, beta_tag, "$(stem)_Estimates.csv"),
    ]

    for path in candidates
        if isfile(path)
            return path
        end
    end

    error(
        "Could not locate estimates for $beta_tag.\nTried:\n" *
        join(["  " * c for c in candidates], "\n")
    )
end


#-------------------------------------------------------------------------------
# load_beta_estimates
#-------------------------------------------------------------------------------
"""
    load_beta_estimates(est_path)

Read one estimation run's point estimates and return `(theta_hat, param_names)`.

The saved CSV holds a single row of parameters. A trailing `NLL` column is
written by some estimation runs and is dropped here so that `theta_hat`
contains only structural parameters, matching how the counterfactual script
consumes the same file.
"""
function load_beta_estimates(est_path::AbstractString)

    df_est = CSV.read(est_path, DataFrame)

    # Drop the objective column when present so only parameters remain
    if "NLL" in names(df_est)
        select!(df_est, Not(:NLL))
    end

    param_names = names(df_est)
    theta_hat   = Float64.(collect(df_est[1, :]))

    return theta_hat, param_names
end


#-------------------------------------------------------------------------------
# compute_hh_loglik
#-------------------------------------------------------------------------------
"""
    compute_hh_loglik(...)

Return a vector of per-household log-likelihood contributions evaluated at a
single specification's point estimates.

This reproduces the mixture log-likelihood exactly as the estimation objective
computes it. For household h with observation range [start, stop]:

    log L_h = logsumexp_k( log pi_k,h + sum_t [ v_k,t(y_t) - logsumexp_j v_k,t(j) ] )

where the type-k mixing weight is a softmax in the household's TYA share and
the choice-specific values are interpolated from the type-k decision value
function at the household's market prices and addiction stocks (bin-specific
prices enter flow utility through the bin ratios, so no correction is needed).

Summing the returned log-likelihood vector reproduces the run's reported
log-likelihood, so the sum is a direct check that the parameters and data line
up.

Returns `(hh_ll, hh_post)`. `hh_post` is an `N_hh x 3` matrix of posterior type
probabilities, each row summing to one, obtained by normalizing the three
mixture terms that the log-likelihood already sums over. These come free with
the likelihood pass and are what the latent-type breakdown classifies on.
"""
function compute_hh_loglik(
    V_decision_1::AbstractArray,
    V_decision_2::AbstractArray,
    V_decision_3::AbstractArray,
    hh_ranges::AbstractVector,
    tya_share_hh::AbstractVector{<:Real},
    tya_state::AbstractVector{<:Integer},
    af_continuous::AbstractVector{<:Real},
    as_continuous::AbstractVector{<:Real},
    aflav_continuous::AbstractVector{<:Real},
    p_continuous::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
    N_J::Integer,
    N_P::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    pi_0_2::Real,
    pi_TYA_2::Real,
    pi_0_3::Real,
    pi_TYA_3::Real
)

    # Number of households
    N_hh = length(hh_ranges)

    # Per-household log-likelihood contributions
    hh_ll = Vector{Float64}(undef, N_hh)

    # Posterior type probabilities, one row per household
    hh_post = Matrix{Float64}(undef, N_hh, 3)

    # Households are independent given parameters, so this loop parallelizes
    Threads.@threads for h in 1:N_hh

        start_idx, stop_idx = hh_ranges[h]

        # Household-specific log mixing weights (softmax with type 1 as base)
        logit_2_h   = pi_0_2 + pi_TYA_2 * tya_share_hh[h]
        logit_3_h   = pi_0_3 + pi_TYA_3 * tya_share_hh[h]
        log_denom_h = logsumexp([0.0, logit_2_h, logit_3_h])
        log_pi_1_h  = -log_denom_h
        log_pi_2_h  = logit_2_h - log_denom_h
        log_pi_3_h  = logit_3_h - log_denom_h

        # Per-type log-likelihood for this household
        log_ll_1 = 0.0
        log_ll_2 = 0.0
        log_ll_3 = 0.0

        for i in start_idx:stop_idx

            # --- Type 1 ---
            v_interp_1 = interpolate_v_choice(
                V_decision_1, tya_state[i], af_continuous[i], as_continuous[i],
                aflav_continuous[i], p_continuous[i, 1], p_continuous[i, 2],
                N_J, N_P, A_f, A_s, A_flav, P
            )
            log_ll_1 += v_interp_1[y[i]] - logsumexp(v_interp_1)

            # --- Type 2 ---
            v_interp_2 = interpolate_v_choice(
                V_decision_2, tya_state[i], af_continuous[i], as_continuous[i],
                aflav_continuous[i], p_continuous[i, 1], p_continuous[i, 2],
                N_J, N_P, A_f, A_s, A_flav, P
            )
            log_ll_2 += v_interp_2[y[i]] - logsumexp(v_interp_2)

            # --- Type 3 ---
            v_interp_3 = interpolate_v_choice(
                V_decision_3, tya_state[i], af_continuous[i], as_continuous[i],
                aflav_continuous[i], p_continuous[i, 1], p_continuous[i, 2],
                N_J, N_P, A_f, A_s, A_flav, P
            )
            log_ll_3 += v_interp_3[y[i]] - logsumexp(v_interp_3)

        end

        # Mixture logsumexp across the three types
        mix_terms = [
            log_pi_1_h + log_ll_1,
            log_pi_2_h + log_ll_2,
            log_pi_3_h + log_ll_3
        ]
        hh_ll[h] = logsumexp(mix_terms)

        # Posterior type probabilities: each mixture term normalized by their
        # logsumexp, which is the household's own log-likelihood
        for k in 1:3
            hh_post[h, k] = exp(mix_terms[k] - hh_ll[h])
        end

    end

    return hh_ll, hh_post
end


#-------------------------------------------------------------------------------
# Standard Normal Reference Distribution
#-------------------------------------------------------------------------------

# The Vuong statistic is asymptotically standard normal under the null, so
# two-sided p-values come from both tails of N(0,1).
const STD_NORMAL = Normal(0.0, 1.0)


#-------------------------------------------------------------------------------
# vuong_test
#-------------------------------------------------------------------------------
"""
    vuong_test(ll_a, ll_b; k_a, k_b, n_obs, label_a, label_b)

Vuong (1989) test for two non-nested models estimated on the same households.

`ll_a` and `ll_b` hold per-household log-likelihood contributions, aligned by
household. The statistic is built from the household-level differences
`d_h = ll_a[h] - ll_b[h]`:

    LR    = sum_h d_h                                     (raw log-likelihood ratio)
    LR_c  = LR - correction                               (complexity-adjusted)
    omega = sample standard deviation of d_h
    V     = LR_c / (sqrt(N) * omega)   ~  N(0, 1) under H0

Under the null the two models are equally close to the true data-generating
process. A large positive `V` favors model A; a large negative `V` favors
model B.

The complexity correction follows Vuong's AIC and BIC variants. Both are
reported, and both are exactly zero when the two models have the same number
of free parameters, which is the case for the beta = 1 and beta = 0.7
specifications compared here since beta is fixed rather than estimated in
each. `n_obs` is the number of household-month choice observations and enters
only the BIC correction.

VARIANCE PRE-TEST. Fixing beta at two different values does not make the two
specifications strictly non-nested: they are OVERLAPPING in Vuong's sense,
since the two families coincide wherever the model collapses to a static one.
With no state dependence in flow utility the continuation value is common
across alternatives, so it drops out of the choice probabilities and beta
stops mattering. Vuong's Section 5 therefore calls for first testing

    H0: omega^2 = 0   (the two models are observationally equivalent here)

Failing to reject leaves the models indistinguishable and the normal test
invalid, because V then divides by something near zero. Rejecting licenses the
directional test above. The pre-test statistic is `N * omega^2`, which is
asymptotically a weighted sum of chi-squares; the exact weights need the
outer-product and Hessian matrices of both models, so rather than approximate
them this function reports `omega^2` with its own sampling standard error and
flags whether it is distinguishable from zero. In this application the two
specifications are far from the static corner, so the pre-test is a formality
that should be reported rather than a binding constraint.

Returns a NamedTuple holding the statistic under each correction along with
the inputs needed to report it.
"""
function vuong_test(
    ll_a::AbstractVector{<:Real},
    ll_b::AbstractVector{<:Real};
    k_a::Integer,
    k_b::Integer,
    n_obs::Integer,
    label_a::AbstractString = "Model A",
    label_b::AbstractString = "Model B"
)

    length(ll_a) == length(ll_b) ||
        error("Per-household log-likelihood vectors have different lengths: $(length(ll_a)) vs $(length(ll_b))")

    # Number of households (the independent sampling unit)
    N = length(ll_a)

    # Household-level log-likelihood differences
    d = collect(Float64, ll_a) .- collect(Float64, ll_b)

    # Total log-likelihoods and the raw likelihood ratio
    total_a = sum(ll_a)
    total_b = sum(ll_b)
    LR      = sum(d)

    # Sample standard deviation of the differences (the variance estimate omega-hat)
    omega = std(d; corrected = true)

    # Variance pre-test for overlapping models: H0 says omega^2 = 0, under which
    # the two specifications are observationally equivalent and the directional
    # test below is invalid. The standard error of a variance estimate is
    # sqrt((mu_4 - omega^4) / N), where mu_4 is the fourth central moment.
    d_centered  = d .- mean(d)
    mu_4        = mean(d_centered .^ 4)
    omega_sq    = omega^2
    omega_sq_se = sqrt(max(mu_4 - omega_sq^2, 0.0) / N)

    # z-statistic for omega^2 = 0 and the scaled pre-test statistic N * omega^2
    omega_sq_z    = omega_sq_se > 0 ? omega_sq / omega_sq_se : NaN
    pretest_stat  = N * omega_sq

    # A z-statistic alone cannot detect the degenerate case: when the two models
    # produce all but identical per-household likelihoods, omega and its standard
    # error both collapse to floating-point noise and their ratio is large but
    # meaningless. Comparing omega against the magnitude of the likelihoods
    # themselves catches that, since omega is measured in the same log-likelihood
    # units. Below this floor the two specifications are treated as
    # observationally equivalent regardless of the z-statistic.
    ll_scale   = (mean(abs.(ll_a)) + mean(abs.(ll_b))) / 2.0
    omega_floor = 1e-8 * max(ll_scale, 1.0)
    degenerate  = omega <= omega_floor

    # The directional test is only interpretable once equivalence is rejected
    overlap_rejected = !degenerate && !isnan(omega_sq_z) && omega_sq_z > 1.96

    # Complexity corrections. Both vanish when k_a == k_b. `n_obs` is the
    # number of independent sampling units (households), matching the BIC in
    # the estimation script.
    corr_aic = (k_a - k_b)
    corr_bic = (k_a - k_b) * log(n_obs) / 2.0

    # Statistic under each correction
    denom = sqrt(N) * omega
    V_raw = denom > 0 ? LR / denom : NaN
    V_aic = denom > 0 ? (LR - corr_aic) / denom : NaN
    V_bic = denom > 0 ? (LR - corr_bic) / denom : NaN

    # Two-sided p-values against the null of equal fit
    p_raw = isnan(V_raw) ? NaN : 2.0 * ccdf(STD_NORMAL, abs(V_raw))
    p_aic = isnan(V_aic) ? NaN : 2.0 * ccdf(STD_NORMAL, abs(V_aic))
    p_bic = isnan(V_bic) ? NaN : 2.0 * ccdf(STD_NORMAL, abs(V_bic))

    # Which model the raw statistic favors at the 5% level. The directional
    # comparison only means something once the variance pre-test has rejected
    # observational equivalence.
    favored = if isnan(V_raw)
        "undetermined"
    elseif !overlap_rejected
        "neither (variance pre-test cannot reject observational equivalence)"
    elseif V_raw > 1.96
        label_a
    elseif V_raw < -1.96
        label_b
    else
        "neither (cannot reject equal fit)"
    end

    return (
        label_a       = label_a,
        label_b       = label_b,
        N_hh          = N,
        N_obs         = n_obs,
        k_a           = k_a,
        k_b           = k_b,
        ll_total_a    = total_a,
        ll_total_b    = total_b,
        LR            = LR,
        mean_diff     = mean(d),
        omega         = omega,
        omega_sq      = omega_sq,
        omega_sq_se   = omega_sq_se,
        omega_sq_z    = omega_sq_z,
        pretest_stat  = pretest_stat,
        degenerate    = degenerate,
        overlap_rejected = overlap_rejected,
        V_raw         = V_raw,
        V_aic         = V_aic,
        V_bic         = V_bic,
        p_raw         = p_raw,
        p_aic         = p_aic,
        p_bic         = p_bic,
        favored       = favored,
        share_a_wins  = mean(d .> 0),
    )
end


#-------------------------------------------------------------------------------
# subgroup_vuong
#-------------------------------------------------------------------------------
"""
    subgroup_vuong(label, idx, ll_a, ll_b; label_a, label_b, N_total)

Recompute the Vuong comparison on the subset of households in `idx`.

Household log-likelihood contributions are independent, so restricting to a
subgroup gives a valid test of equal fit within that subpopulation. The
statistic is built exactly as in `vuong_test`, from the subgroup's own mean and
standard deviation of the differences.

Two summaries are reported side by side because they answer different
questions and can disagree sharply:

  * `mean_diff` drives the Vuong statistic. It is the average household's
    advantage, so a small number of households with very large differences can
    determine its sign.
  * `median_diff` and `share_a_wins` describe the TYPICAL household. When most
    households mildly prefer one model while a minority strongly prefers the
    other, these point the opposite way from the mean.

The quartiles are reported so that this skew is visible rather than inferred.

Returns a NamedTuple with one row's worth of results.
"""
function subgroup_vuong(
    label::AbstractString,
    idx::AbstractVector{<:Integer},
    ll_a::AbstractVector{<:Real},
    ll_b::AbstractVector{<:Real};
    label_a::AbstractString = "Model A",
    label_b::AbstractString = "Model B",
    N_total::Integer = length(ll_a)
)

    # Subgroup log-likelihoods and their differences
    a = Float64.(ll_a[idx])
    b = Float64.(ll_b[idx])
    d = a .- b
    N = length(d)

    # Return an empty row if the subgroup has no households
    if N == 0
        return (
            group = label, N_hh = 0, share_of_hh = 0.0,
            ll_total_a = 0.0, ll_total_b = 0.0, LR = 0.0,
            mean_diff = NaN, median_diff = NaN, q25_diff = NaN, q75_diff = NaN,
            omega = NaN, V = NaN, p = NaN, share_a_wins = NaN,
            favored = "empty",
        )
    end

    LR    = sum(d)
    omega = N > 1 ? std(d; corrected = true) : NaN
    denom = sqrt(N) * omega

    V = (N > 1 && denom > 0) ? LR / denom : NaN
    p = isnan(V) ? NaN : 2.0 * ccdf(STD_NORMAL, abs(V))

    return (
        group        = label,
        N_hh         = N,
        share_of_hh  = N / N_total,
        ll_total_a   = sum(a),
        ll_total_b   = sum(b),
        LR           = LR,
        mean_diff    = mean(d),
        median_diff  = median(d),
        q25_diff     = quantile(d, 0.25),
        q75_diff     = quantile(d, 0.75),
        omega        = omega,
        V            = V,
        p            = p,
        share_a_wins = mean(d .> 0),
        favored      = isnan(V) ? "undetermined" :
                       (V >  1.96 ? label_a :
                       (V < -1.96 ? label_b : "neither")),
    )
end


#-------------------------------------------------------------------------------
# log_subgroup_table
#-------------------------------------------------------------------------------
"""
    log_subgroup_table(rows, label_a, label_b)

Write a set of `subgroup_vuong` rows to the log in a readable block.

The mean and the median are printed together, since a sign disagreement
between them is the substantive finding whenever it occurs.
"""
function log_subgroup_table(
    rows::AbstractVector,
    label_a::AbstractString,
    label_b::AbstractString
)

    for r in rows

        log_msg(@sprintf("%-22s  N = %6d (%5.1f%% of households)",
            r.group, r.N_hh, 100 * r.share_of_hh))

        r.N_hh == 0 && continue

        log_msg(@sprintf("    LR (A - B) = %10.2f    omega = %7.5f", r.LR, r.omega))
        log_msg(@sprintf("    mean diff  = %8.5f    median diff = %8.5f    [q25, q75] = [%7.4f, %7.4f]",
            r.mean_diff, r.median_diff, r.q25_diff, r.q75_diff))
        log_msg(@sprintf("    V = %8.3f    p = %.3e    share favoring %s = %.4f",
            r.V, r.p, label_a, r.share_a_wins))

        # Flag the case where the average household and the typical household
        # disagree, which the headline statistic alone would hide
        if !isnan(r.mean_diff) && !isnan(r.median_diff) && r.median_diff != 0.0 && sign(r.mean_diff) != sign(r.median_diff)
            log_msg(@sprintf("    NOTE: mean and median disagree: the average favors %s but the typical household favors %s",
                r.mean_diff > 0 ? label_a : label_b,
                r.median_diff > 0 ? label_a : label_b))
        end

        log_msg(@sprintf("    Favored (mean criterion): %s", r.favored))
        log_msg("")

    end
end
