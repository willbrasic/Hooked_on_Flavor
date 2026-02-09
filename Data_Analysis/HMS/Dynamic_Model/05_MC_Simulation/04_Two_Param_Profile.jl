################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Two-Parameter Profile Likelihood Scan
#
# Evaluates the NLL over a 2D grid of two parameters while holding all
# other parameters at their true values. This diagnoses ridges and
# multimodality in the likelihood surface.
#
# Default: profiles (α_T, γ) — the pair with the largest bias in MC results.
# Change param_idx_1 and param_idx_2 to profile other pairs.
#
# Usage:
#   julia 04_Two_Param_Profile.jl
################################################################################


#############################
# Preliminaries
#############################

hpc = !Sys.iswindows()

if hpc
    include("../01_Functions.jl")
    include("01_MC_Simulation_Functions.jl")
    cd("../../Data")
else
    include("../02_Second_Stage_Estimation/01_Functions.jl")
    include("01_MC_Simulation_Functions.jl")
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

using Random
Random.seed!(42)

# Route logging to stdout only
global est_log_io = nothing
global mc_log_io  = nothing


#############################
# Load Data
#############################

println("Loading data...")
t_start = time()

_, β, δ = get_fixed_parameters()
# Addiction grid created below using ψ from θ_true
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

# Real household data for design-based simulation
real_hh_codes = get_hh_codes()
_, real_tya = get_teen_young_adult()
real_tya_state = get_tya_state(real_tya)
_, real_p_continuous = map_prices_to_grid(N_P, P, Pcomb)

println("Data loaded in $(round(time() - t_start, digits=1))s")


#############################
# True Parameters
#############################

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
    ξ_TE = -6.05,
    ψ    =  0.94
)
θ_true_vec = collect(Float64, values(θ_true))
param_names = collect(String, string.(keys(θ_true)))
N_params = length(θ_true)


#############################
# Solve DGP Value Function
#############################

println("\nSolving VFI at true parameters...")
t_vfi = time()

# Create addiction grid using the true ψ
ψ_true = θ_true.ψ
N_A, A = get_addiction_space(ψ_true)

U_true = get_flow_utility(
    θ_true_vec[1:end-1], N_J, N_A, N_Pcomb, A, c_cig, c_ecig, c_bundle, n, is_flavored, cat_idx, E
)
a_lower, a_upper, a_weight = precompute_addiction_transitions(N_J, N_A, ψ_true, A, n)
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

println("\nSimulating data...")
t_sim = time()

global y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim
y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim = simulate_data(
    V_decision_true, ψ_true, N_J, N_P, A, P, n,
    real_p_continuous, real_tya_state, real_hh_codes
)
println("Simulated $(length(y_sim)) observations in $(round(time()-t_sim, digits=1))s")


#############################
# Profile Configuration
#############################

# Which two parameters to profile (change these to test other pairs)
param_idx_1 = 1    # α_T
param_idx_2 = 7    # γ

name_1 = param_names[param_idx_1]
name_2 = param_names[param_idx_2]
true_1 = θ_true_vec[param_idx_1]
true_2 = θ_true_vec[param_idx_2]

# Grid ranges: scan from 20% to 300% of true value (or wider for small values)
# For parameters near zero, use absolute offsets instead
function make_grid(true_val, N_pts)
    if abs(true_val) < 0.1
        # Small parameter: scan ± 1.0 around truth
        lo = true_val - 1.0
        hi = true_val + 1.0
    else
        # Larger parameter: scan 20% to 300% of true
        if true_val > 0
            lo = true_val * 0.2
            hi = true_val * 3.0
        else
            lo = true_val * 3.0   # more negative
            hi = true_val * 0.2   # less negative
        end
    end
    return collect(range(lo, hi, length=N_pts))
end

N_grid = 15    # points per dimension (total = N_grid^2 VFI solves)
grid_1 = make_grid(true_1, N_grid)
grid_2 = make_grid(true_2, N_grid)

println("\n" * "="^60)
println("TWO-PARAMETER PROFILE: ($name_1, $name_2)")
println("  $name_1: $(round(grid_1[1], digits=4)) to $(round(grid_1[end], digits=4)) ($N_grid points, true=$true_1)")
println("  $name_2: $(round(grid_2[1], digits=4)) to $(round(grid_2[end], digits=4)) ($N_grid points, true=$true_2)")
println("  Total evaluations: $(N_grid^2)")
println("="^60)


#############################
# 2D Grid Evaluation
#############################

nll_grid = Matrix{Float64}(undef, N_grid, N_grid)
eval_count_profile = 0
t_grid = time()

for i in 1:N_grid
    for j in 1:N_grid
        eval_count_profile += 1

        # Build full θ with only target parameters changed
        θ_full = copy(θ_true_vec)
        θ_full[param_idx_1] = grid_1[i]
        θ_full[param_idx_2] = grid_2[j]

        # Extract ψ from θ_full (last element) and recompute addiction grid
        ψ_current = θ_full[end]
        N_A_current, A_current = get_addiction_space(ψ_current)

        # Flow utility at candidate θ (first 11 elements, excluding ψ)
        U_current = get_flow_utility(
            θ_full[1:end-1], N_J, N_A_current, N_Pcomb, A_current, c_cig, c_ecig, c_bundle,
            n, is_flavored, cat_idx, E
        )

        # Addiction transitions at candidate ψ
        a_lower_current, a_upper_current, a_weight_current = precompute_addiction_transitions(
            N_J, N_A_current, ψ_current, A_current, n
        )

        # VFI
        _, V_decision_current, iters, converged = solve_vfi(
            N_J, N_A_current, N_P, N_Pcomb, β, δ, U_current,
            a_lower_current, a_upper_current, a_weight_current,
            p_cig_lo, p_cig_hi, p_cig_w,
            p_ecig_lo, p_ecig_hi, p_ecig_w
        )

        if !converged
            nll_grid[i, j] = 1e14
            println(@sprintf("  [%3d/%d] PENALTY (VFI not converged) | %s=%.4f, %s=%.4f",
                eval_count_profile, N_grid^2, name_1, grid_1[i], name_2, grid_2[j]))
            continue
        end

        # Addiction trajectories at candidate ψ
        a0_current, _ = get_initial_addiction_stock_mc(ψ_current, A_current, n, y_sim, hh_codes_sim)
        _, a_continuous_current = simulate_addiction_trajectories_mc(
            N_A_current, ψ_current, A_current, n, y_sim, a0_current, hh_codes_sim
        )

        # Log-likelihood
        LL = log_likelihood(
            V_decision_current, N_J, N_P, A_current, P,
            y_sim, tya_state_sim, a_continuous_current, p_continuous_sim
        )

        nll_grid[i, j] = -LL
        println(@sprintf("  [%3d/%d] NLL = %10.4f | VFI iters = %3d | %s=%.4f, %s=%.4f",
            eval_count_profile, N_grid^2, -LL, iters, name_1, grid_1[i], name_2, grid_2[j]))
    end
end

grid_elapsed = time() - t_grid
println("\nGrid evaluation complete: $(N_grid^2) points in $(round(grid_elapsed, digits=1))s")


#############################
# Find Minimum
#############################

# Find global minimum on the grid
best_idx = argmin(nll_grid)
best_i, best_j = best_idx[1], best_idx[2]
best_val_1 = grid_1[best_i]
best_val_2 = grid_2[best_j]
best_nll = nll_grid[best_i, best_j]

# NLL at true values (find closest grid point)
true_i = argmin(abs.(grid_1 .- true_1))
true_j = argmin(abs.(grid_2 .- true_2))
true_nll = nll_grid[true_i, true_j]

println("\n" * "="^60)
println("RESULTS")
println("="^60)
println(@sprintf("  Grid minimum:  %s=%.4f, %s=%.4f  (NLL=%.4f)", name_1, best_val_1, name_2, best_val_2, best_nll))
println(@sprintf("  True values:   %s=%.4f, %s=%.4f  (NLL=%.4f)", name_1, true_1, name_2, true_2, true_nll))
println(@sprintf("  NLL difference: %.4f", true_nll - best_nll))


#############################
# Save Grid to CSV
#############################

if hpc
    output_dir = abspath("../MC_Simulation_Results")
else
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/MC_Simulation_Results"
end

# Save as long-format CSV for easy plotting
csv_path = joinpath(output_dir, "Profile_$(name_1)_$(name_2).csv")
open(csv_path, "w") do io
    println(io, "$name_1,$name_2,NLL")
    for i in 1:N_grid
        for j in 1:N_grid
            println(io, @sprintf("%.6f,%.6f,%.6f", grid_1[i], grid_2[j], nll_grid[i, j]))
        end
    end
end
println("\nGrid saved to: $csv_path")

# Print NLL table for quick visual inspection
println("\nNLL grid ($name_1 across rows, $name_2 across columns):")
println("  NLL values relative to minimum (0 = best)")
println()

# Print column headers (γ values)
col_indices = 1:3:N_grid  # show every 3rd column to fit
print(@sprintf("  %10s", "$name_1 \\ $name_2"))
for j in col_indices
    print(@sprintf("  %10.4f", grid_2[j]))
end
println()
print("  " * repeat("-", 10))
for _ in col_indices
    print("  " * repeat("-", 10))
end
println()

# Print rows (α_T values)
for i in 1:N_grid
    print(@sprintf("  %10.4f", grid_1[i]))
    for j in col_indices
        rel_nll = nll_grid[i, j] - best_nll
        if rel_nll > 1e10
            print(@sprintf("  %10s", "PENALTY"))
        else
            print(@sprintf("  %10.2f", rel_nll))
        end
    end
    if abs(grid_1[i] - true_1) < (grid_1[2] - grid_1[1]) * 0.6
        print("  ← true $name_1")
    end
    println()
end

println("\n  ↑ columns near true $name_2 = $true_2")
println("\nTotal elapsed: $(round(time() - t_start, digits=1))s")
