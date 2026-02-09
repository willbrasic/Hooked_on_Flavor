################################################################################
# Single-Parameter Recovery Test
#
# Tests whether the estimation machinery can recover a single parameter
# while holding all others fixed at their true values. Start with α_T,
# then expand to more parameters.
################################################################################


#############################
# Preliminaries
#############################

hpc = false

if hpc
    include("../01_Functions.jl")
    include("01_MC_Simulation_Functions.jl")
    cd("../../Data")
else
    include("../02_Second_Stage_Estimation/01_Functions.jl")
    include("01_MC_Simulation_Functions.jl")
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

# Route logging to stdout only
global est_log_io = nothing
global mc_log_io  = nothing


#############################
# Load Data
#############################

println("Loading data...")
t_start = time()

ψ, β, δ = get_fixed_parameters()
N_A, A = get_addiction_space(ψ)
_, N_J, J = get_product_choices()
N_K, _ = get_category_choices()

N_cig, N_orig_ecig, N_flav_ecig, _, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J)
n, n_max = get_nicotine(N_J)
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_flav_ecig)
is_flavored = get_flavored_indicator(cat_idx)

N_P, P = get_pricing_spaces()
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P)
E, E_max = get_expenditures(N_J, N_Pcomb, c_cig, c_ecig, c_cig_max, c_ecig_max, Pcomb)
T = get_transitions(N_K)
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T)

# AR(1) parameters
AR_Phi = CSV.read("../AR_Parameters/AR_Parameters_Phi.csv", DataFrame)
φ_0 = [AR_Phi[1, :intercept], AR_Phi[2, :intercept]]
φ_1 = [AR_Phi[1, :ar1], AR_Phi[2, :ar1]]
AR_Sigma = CSV.read("../AR_Parameters/AR_Parameters_Sigma.csv", DataFrame)
Σ = [AR_Sigma[1, :cig] AR_Sigma[1, :ecig]; AR_Sigma[2, :cig] AR_Sigma[2, :ecig]]
L_chol = cholesky(Σ).L

println("Data loaded in $(round(time() - t_start, digits=1))s")


#############################
# True Parameters
#############################

# Realistic true DGP — close to static logit estimates so that
# the starting values used in actual estimation are in the right ballpark.
# α_TE set to 0.50 (not 9.57 from static logit) to avoid bundle dominance.
θ_true = (
    α_T  =  0.46,
    α_E  =  0.37,
    α_TE =  0.50,
    λ_1  =  0.67,
    λ_2  =  0.41,
    μ    =  0.05,
    γ    = -0.05,
    ω    = -1.94,
    ξ_T  = -3.61,
    ξ_E  = -5.46,
    ξ_TE = -6.05
)
θ_true_vec = collect(Float64, values(θ_true))
param_names = collect(String, string.(keys(θ_true)))
N_params = length(θ_true)

println("\nTrue parameters:")
for (k, v) in pairs(θ_true)
    println("  $k = $v")
end


#############################
# Solve DGP Value Function
#############################

println("\nSolving VFI at true parameters...")
t_vfi = time()

U_true = get_flow_utility(
    θ_true_vec, N_J, N_A, N_Pcomb, A, c_cig, c_ecig, c_bundle, n, is_flavored, cat_idx, E
)

# Pre-compute addiction transitions (ψ is fixed, so compute once)
a_lower, a_upper, a_weight = precompute_addiction_transitions(N_J, N_A, ψ, A, n)

_, V_decision_true, vfi_iters, vfi_converged = solve_vfi(
    N_J, N_A, N_P, N_Pcomb, β, δ, U_true,
    a_lower, a_upper, a_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w
)
println("DGP VFI: $vfi_iters iters, converged=$vfi_converged ($(round(time()-t_vfi, digits=1))s)")


#############################
# Simulate Data
#############################

N_HH_sim = 200
T_sim    = 24
N_obs    = N_HH_sim * T_sim

println("\nSimulating data: $N_HH_sim HHs × $T_sim periods = $N_obs obs")
t_sim = time()

global y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim
y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim = simulate_data(
    V_decision_true, N_HH_sim, T_sim, ψ, N_J, N_P, A, P, n,
    φ_0, φ_1, L_chol
)
println("Simulation complete in $(round(time()-t_sim, digits=1))s")

# Check simulated choice distribution
println("\nSimulated category shares:")
cat_labels = ["Outside", "Cig", "OrgEcig", "FlvEcig", "OrgBundle", "FlvBundle"]
for k in 0:5
    share = count(cat_idx[y_sim[i]] == k for i in 1:length(y_sim)) / length(y_sim)
    println(@sprintf("  %-10s  %6.2f%%", cat_labels[k+1], share * 100))
end


#############################
# Single-Parameter Objective
#############################

# Which parameter to test (change this to test different params)
param_idx  = 1    # 1 = α_T
param_name = param_names[param_idx]
true_val   = θ_true_vec[param_idx]

println("\n" * "="^60)
println("SINGLE-PARAMETER TEST: $param_name")
println("True value: $true_val")
println("="^60)

# Objective: varies only the target parameter, holds rest at truth
eval_count_single = 0

function objective_single(x::AbstractVector{<:Real})
    global eval_count_single
    eval_count_single += 1

    # Build full θ with only target parameter changed
    θ_full = copy(θ_true_vec)
    θ_full[param_idx] = x[1]

    # Flow utility at candidate θ
    U_current = get_flow_utility(
        θ_full, N_J, N_A, N_Pcomb, A, c_cig, c_ecig, c_bundle,
        n, is_flavored, cat_idx, E
    )

    # VFI (addiction transitions unchanged since ψ is fixed)
    _, V_decision_current, iters, converged = solve_vfi(
        N_J, N_A, N_P, N_Pcomb, β, δ, U_current,
        a_lower, a_upper, a_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w
    )

    if !converged
        println(@sprintf("  Eval %d | PENALTY (VFI not converged) | %s = %.6f",
            eval_count_single, param_name, x[1]))
        return 1e14
    end

    # Addiction trajectories for simulated data
    a0_current, _ = get_initial_addiction_stock_mc(ψ, A, n, y_sim, hh_codes_sim)
    _, a_continuous_current = simulate_addiction_trajectories_mc(
        N_A, ψ, A, n, y_sim, a0_current, hh_codes_sim
    )

    # Log-likelihood
    LL = log_likelihood(
        V_decision_current, N_J, N_P, A, P,
        y_sim, tya_state_sim, a_continuous_current, p_continuous_sim
    )

    nll = -LL
    println(@sprintf("  Eval %d | NLL = %.4f | VFI iters = %d | %s = %.6f",
        eval_count_single, nll, iters, param_name, x[1]))

    return nll
end


#############################
# Grid Search
#############################

println("\n--- Grid Search over $param_name ---")
eval_count_single = 0

# Search from 20% to 300% of true value
grid_lo  = true_val * 0.2
grid_hi  = true_val * 3.0
N_grid   = 11
grid_vals = collect(range(grid_lo, grid_hi, length=N_grid))

grid_nll = Float64[]
t_grid = time()
for val in grid_vals
    nll = objective_single([val])
    push!(grid_nll, nll)
end
grid_elapsed = time() - t_grid

# Find grid minimum
best_grid_idx = argmin(grid_nll)
best_grid_val = grid_vals[best_grid_idx]

println("\nGrid search results ($N_grid points, $(round(grid_elapsed, digits=1))s):")
println(@sprintf("  %-14s  %12s  %s", param_name, "NLL", ""))
println("  " * repeat("-", 40))
for i in 1:N_grid
    marker = (i == best_grid_idx) ? " ← grid min" : ""
    if abs(grid_vals[i] - true_val) < (grid_hi - grid_lo) / (2 * N_grid)
        marker *= " (≈ true)"
    end
    println(@sprintf("  %-14.6f  %12.4f%s", grid_vals[i], grid_nll[i], marker))
end
println("\nGrid minimum: $param_name = $(round(best_grid_val, digits=6))")
println("True value:   $param_name = $true_val")


#############################
# Nelder-Mead Optimization
#############################

println("\n--- Nelder-Mead Optimization ---")
eval_count_single = 0

# Start from the grid minimum
t_opt = time()
result = optimize(
    objective_single, [best_grid_val],
    NelderMead(initial_simplex = SimplexWithAdd([abs(true_val) * 0.3])),
    Optim.Options(iterations = 200, f_abstol = 1e-4)
)
opt_elapsed = time() - t_opt

opt_val = Optim.minimizer(result)[1]
opt_nll = Optim.minimum(result)

println("\nOptimization results ($(round(opt_elapsed, digits=1))s):")
println("  Converged:    $(Optim.converged(result))")
println("  Iterations:   $(Optim.iterations(result))")
println("  Evaluations:  $eval_count_single")
println("  NLL at opt:   $(round(opt_nll, digits=4))")
println()
println("  $param_name estimated: $(round(opt_val, digits=6))")
println("  $param_name true:      $true_val")
println("  Difference:           $(round(opt_val - true_val, digits=6))")
println("  % Error:              $(round(100 * (opt_val - true_val) / true_val, digits=2))%")

println("\nTotal elapsed: $(round(time() - t_start, digits=1))s")
