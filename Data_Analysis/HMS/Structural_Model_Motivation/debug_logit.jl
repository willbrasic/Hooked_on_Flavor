# Diagnostic script: compare L-BFGS and Amoeba solutions
# Evaluates nll, gradient, and total effects at both optima

# Load functions and data (same setup as 03_Static_Logit.jl)
using Dates
include("../Dynamic_Model/02_Second_Stage_Estimation/01_Functions.jl")
cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")

# --- Data loading (identical to 03_Static_Logit.jl) ---
_, N_J, J = get_product_choices()
y = get_hh_choices(J)
N_K, _ = get_category_choices()
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, c_cig_std, c_ecig_std, c_bundle_std, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J)
c_cig    = c_cig_std    .* c_cig_max
c_ecig   = c_ecig_std   .* c_ecig_max
c_bundle = c_bundle_std .* c_bundle_max
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig)
is_non_fda_flavored = get_non_fda_flavored_indicator(cat_idx)
is_fda_flavored = get_fda_flavored_indicator(cat_idx)
ratio_cig, ratio_ecig = get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, c_cig_std, c_ecig_std)
_, tya = get_teen_young_adult()
tya_state = get_tya_state(tya)
N_P, P = get_pricing_spaces()
_, Pcomb = get_pricing_spaces_combination(N_K, N_P, P)
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb)
df_lag = CSV.read("./Lagged_Category_Choice.csv", DataFrame)
lag_cig_raw      = df_lag.lagged_cig
lag_ecig_raw     = df_lag.lagged_ecig
lag_cig_ecig_raw = df_lag.lagged_cig_ecig
valid = .!ismissing.(lag_cig_raw)
valid_idx = findall(valid)
y            = y[valid_idx]
tya_state    = tya_state[valid_idx]
p_continuous = p_continuous[valid_idx, :]
lag_cig      = Int64.(collect(skipmissing(lag_cig_raw[valid_idx])))
lag_ecig     = Int64.(collect(skipmissing(lag_ecig_raw[valid_idx])))
lag_cig_ecig = Int64.(collect(skipmissing(lag_cig_ecig_raw[valid_idx])))
N_obs = length(y)

# Pre-compute
fe_idx = zeros(Int, N_J)
for j in 1:N_J
    if cat_idx[j] == 1
        fe_idx[j] = 1
    elseif cat_idx[j] in (2, 3, 4)
        fe_idx[j] = 2
    elseif cat_idx[j] in (5, 6, 7)
        fe_idx[j] = 3
    end
end
fe_T  = fe_idx .== 1
fe_E  = fe_idx .== 2
fe_TE = fe_idx .== 3
tya = [s == 2 ? 1 : 0 for s in tya_state]
E_obs = p_continuous[:, 1] * (ratio_cig .* c_cig)' + p_continuous[:, 2] * (ratio_ecig .* c_ecig)'
lag_match = zeros(Int64, N_obs, N_J)
for i in 1:N_obs
    for j in 1:N_J
        if (fe_idx[j] == 1 && lag_cig[i] == 1) ||
           (fe_idx[j] == 2 && lag_ecig[i] == 1) ||
           (fe_idx[j] == 3 && lag_cig_ecig[i] == 1)
            lag_match[i, j] = 1
        end
    end
end

# --- neg_log_likelihood (same as 03_Static_Logit.jl) ---
function neg_log_likelihood(θ_vec, N_obs, N_J, tya, y,
                            c_cig, c_ecig, c_bundle,
                            is_non_fda_flavored, is_fda_flavored,
                            lag_match, E_obs,
                            fe_T, fe_E, fe_TE)
    α_T  = θ_vec[1];  α_E  = θ_vec[2];  α_TE = θ_vec[3]
    λ_1  = θ_vec[4];  λ_2  = θ_vec[5]
    λ_3  = θ_vec[6];  λ_4  = θ_vec[7]
    ρ    = θ_vec[8];  ω    = θ_vec[9]
    ξ_T  = θ_vec[10]; ξ_E  = θ_vec[11]; ξ_TE = θ_vec[12]

    neg_LL = zero(eltype(θ_vec))
    v = Vector{eltype(θ_vec)}(undef, N_J)
    for i in 1:N_obs
        tya_i = tya[i]; y_i = y[i]
        for j in 1:N_J
            v[j] = (α_T * c_cig[j] + α_E * c_ecig[j] + α_TE * c_bundle[j]
                   + is_non_fda_flavored[j] * (λ_1 + λ_2 * tya_i)
                   + is_fda_flavored[j] * (λ_3 + λ_4 * tya_i)
                   + ρ * lag_match[i, j]
                   + ω * E_obs[i, j]
                   + ξ_T * fe_T[j] + ξ_E * fe_E[j] + ξ_TE * fe_TE[j])
        end
        m = maximum(v)
        s = zero(eltype(v))
        for j in 1:N_J; s += exp(v[j] - m); end
        neg_LL -= v[y_i] - m - log(s)
    end
    return neg_LL
end

nll = θ -> neg_log_likelihood(θ, N_obs, N_J, tya, y,
                              c_cig, c_ecig, c_bundle,
                              is_non_fda_flavored, is_fda_flavored,
                              lag_match, E_obs,
                              fe_T, fe_E, fe_TE)

# --- Two solutions to compare ---
# L-BFGS result (run 1, from zeros)
θ_lbfgs = [0.017878, 0.009934, -0.000681, -0.271808, 0.364739,
           -0.932604, -0.001731, 2.756677, -0.005350, -3.567006, -5.898830, -5.215384]

# Amoeba result (run 1)
θ_amoeba = [0.017975, 0.008915, -0.001581, 0.177776, 0.555524,
            0.080213, 0.219706, 2.765269, -0.005356, -3.573967, -6.388166, -5.387856]

# --- Evaluate nll ---
println("=" ^ 60)
println("NLL COMPARISON")
println("=" ^ 60)
nll_lbfgs  = nll(θ_lbfgs)
nll_amoeba = nll(θ_amoeba)
println("L-BFGS neg LL:  $(round(nll_lbfgs, digits=4))")
println("Amoeba neg LL:  $(round(nll_amoeba, digits=4))")
println("Difference:     $(round(nll_lbfgs - nll_amoeba, digits=4))")

# --- Gradients ---
println("\n" * "=" ^ 60)
println("GRADIENT COMPARISON")
println("=" ^ 60)
param_names = ["α_T", "α_E", "α_TE", "λ_1", "λ_2", "λ_3", "λ_4", "ρ", "ω", "ξ_T", "ξ_E", "ξ_TE"]

using ForwardDiff
grad_lbfgs  = ForwardDiff.gradient(nll, θ_lbfgs)
grad_amoeba = ForwardDiff.gradient(nll, θ_amoeba)

println("\nGradient at L-BFGS solution:")
for (i, name) in enumerate(param_names)
    @Printf.printf("  %-6s  %12.4f\n", name, grad_lbfgs[i])
end
println("  |∇|  = $(round(sqrt(sum(grad_lbfgs.^2)), digits=6))")

println("\nGradient at Amoeba solution:")
for (i, name) in enumerate(param_names)
    @Printf.printf("  %-6s  %12.4f\n", name, grad_amoeba[i])
end
println("  |∇|  = $(round(sqrt(sum(grad_amoeba.^2)), digits=6))")

# --- Total effects (decomposition check) ---
println("\n" * "=" ^ 60)
println("TOTAL UTILITY EFFECTS (intercept part only, no TYA)")
println("=" ^ 60)
println("\n                         L-BFGS       Amoeba")
println("  Orig ecig (ξ_E):     $(round(θ_lbfgs[11], digits=4))    $(round(θ_amoeba[11], digits=4))")
println("  NonFDA flav (ξ_E+λ1):$(round(θ_lbfgs[11]+θ_lbfgs[4], digits=4))    $(round(θ_amoeba[11]+θ_amoeba[4], digits=4))")
println("  FDA flav (ξ_E+λ3):   $(round(θ_lbfgs[11]+θ_lbfgs[6], digits=4))    $(round(θ_amoeba[11]+θ_amoeba[6], digits=4))")

println("\nTOTAL UTILITY EFFECTS (with TYA)")
println("  NonFDA (ξ_E+λ1+λ2):  $(round(θ_lbfgs[11]+θ_lbfgs[4]+θ_lbfgs[5], digits=4))    $(round(θ_amoeba[11]+θ_amoeba[4]+θ_amoeba[5], digits=4))")
println("  FDA (ξ_E+λ3+λ4):     $(round(θ_lbfgs[11]+θ_lbfgs[6]+θ_lbfgs[7], digits=4))    $(round(θ_amoeba[11]+θ_amoeba[6]+θ_amoeba[7], digits=4))")

# --- Choice shares by category ---
println("\n" * "=" ^ 60)
println("CHOICE SHARES BY CATEGORY")
println("=" ^ 60)
for (cat, label) in [(0, "Outside"), (1, "Cig"), (2, "Orig ecig"),
                      (3, "NonFDA flav ecig"), (4, "FDA flav ecig"),
                      (5, "Bundle orig"), (6, "Bundle NonFDA"), (7, "Bundle FDA")]
    alts = findall(cat_idx .== cat)
    share = sum(count(y .== j) for j in alts) / N_obs * 100
    @Printf.printf("  cat %d %-20s  %6.2f%% (%d alts)\n", cat, label, share, length(alts))
end

# --- TYA breakdown ---
n_tya = sum(tya .== 1)
n_notya = sum(tya .== 0)
println("\nTYA breakdown: $(n_tya) TYA ($(round(n_tya/N_obs*100, digits=1))%), $(n_notya) non-TYA")
