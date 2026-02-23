# Full-Solution GMM Estimation Notes

Notes on reformulating the dynamic model from MLE to full-solution GMM. The VFI machinery is unchanged — only the criterion function, moment construction, and standard errors differ.

## Why Full-Solution GMM?

MLE assumes the logit error structure (Type I EV) is correctly specified. If the error distribution is misspecified, MLE is inconsistent. Full-solution GMM with appropriate moment conditions can be consistent under weaker distributional assumptions — it only requires that the moment conditions hold in expectation, not that the full likelihood is correct.

Trade-off: GMM is less efficient than MLE when the model is correctly specified (MLE achieves the Cramér-Rao bound). The computational cost is similar since VFI is still solved at every parameter evaluation.

## What Changes

### Current MLE Criterion
```
max_θ  ℓ(θ) = Σᵢ log P(yᵢ | xᵢ; θ)

where P(j | xᵢ; θ) = exp(V_d[j, xᵢ; θ]) / Σ_k exp(V_d[k, xᵢ; θ])
```

### GMM Criterion
```
min_θ  Q(θ) = ḡ(θ)' W ḡ(θ)

where ḡ(θ) = (1/N) Σᵢ g(yᵢ, xᵢ; θ)
      g(yᵢ, xᵢ; θ) = (𝟙[yᵢ = j] - P(j | xᵢ; θ)) ⊗ z(xᵢ)
```

The residual `e_ij(θ) = 𝟙[yᵢ = j] - P(j | xᵢ; θ)` is the prediction error: 1 minus the predicted probability for the chosen alternative, and 0 minus the predicted probability for all others. The moment condition E[e_ij · z_i] = 0 says these residuals should be uncorrelated with instruments z.

### Standard Errors
```
MLE:  V̂(θ) = H⁻¹                         (inverse Hessian of log-likelihood)
GMM:  V̂(θ) = (G'WG)⁻¹ G'W Ω̂ WG (G'WG)⁻¹  (sandwich formula)

where G = (1/N) Σᵢ ∂g(yᵢ, xᵢ; θ)/∂θ'    (Jacobian of moments)
      Ω̂ = (1/N) Σᵢ g(yᵢ, xᵢ; θ̂) g(yᵢ, xᵢ; θ̂)'  (variance of moments)
      W = weighting matrix
```

With optimal weighting W = Ω̂⁻¹ (two-step GMM), this simplifies to V̂(θ) = (G'Ω̂⁻¹G)⁻¹.

## Moment Conditions

### Baseline Moments: Aggregate Share Matching
The simplest specification matches predicted to observed choice shares:
```
g₁(θ) = (1/N) Σᵢ [𝟙[yᵢ = j] - P(j | xᵢ; θ)]     for j = 1, ..., 40
```
This gives 40 moment conditions (one per alternative). With 13 parameters, the model is overidentified (27 overidentifying restrictions). The J-test (Hansen's test of overidentifying restrictions) can be used as a specification test.

### Richer Moments: Interactions with State Variables
Interact residuals with observed state variables for more informative moments:
```
g₂(θ) = (1/N) Σᵢ [𝟙[yᵢ = j] - P(j | xᵢ; θ)] · tya_i          (TYA interactions)
g₃(θ) = (1/N) Σᵢ [𝟙[yᵢ = j] - P(j | xᵢ; θ)] · ã_i            (addiction interactions)
g₄(θ) = (1/N) Σᵢ [𝟙[yᵢ = j] - P(j | xᵢ; θ)] · p_cig_i        (cig price interactions)
g₅(θ) = (1/N) Σᵢ [𝟙[yᵢ = j] - P(j | xᵢ; θ)] · p_ecig_i       (ecig price interactions)
```

These interact the prediction error with each dimension of the state, giving up to 40 × 5 = 200 moments. In practice, aggregate to category-level residuals (8 categories × 5 instruments = 40 moments) to avoid a near-singular moment variance matrix.

### Category-Level Moment Specification (Recommended Starting Point)
Aggregate alternatives within each category k ∈ {0, 1, ..., 7}:
```
e_ik(θ) = 𝟙[yᵢ ∈ cat k] - Σ_{j ∈ cat k} P(j | xᵢ; θ)

Moments:
(1/N) Σᵢ e_ik · 1           = 0     (8 moments: mean category shares)
(1/N) Σᵢ e_ik · tya_i       = 0     (8 moments: TYA interactions)
(1/N) Σᵢ e_ik · ã_i         = 0     (8 moments: addiction interactions)
(1/N) Σᵢ e_ik · p_cig_i     = 0     (8 moments: cig price interactions)
(1/N) Σᵢ e_ik · p_ecig_i    = 0     (8 moments: ecig price interactions)
```
Total: 40 moments, 13 parameters → 27 overidentifying restrictions.

## Instruments for Endogeneity

### Consumption Endogeneity

The addiction stock ã_i is constructed from past choices, which creates a mechanical endogeneity: unobserved taste shocks that drove past consumption also drive current consumption, and past consumption determines ã. Instruments for ã should be correlated with the addiction stock but uncorrelated with current taste shocks.

**Candidate instruments for addiction (ã):**

1. **Lagged prices (p_{i,t-L} for L ≥ 2).** Past prices shift past consumption (and hence ã) but should be uncorrelated with current ε if prices follow the AR(1) and shocks are serially uncorrelated. Use L ≥ 2 to avoid overlap with the AR(1) innovation structure. The exclusion restriction requires that lagged prices affect current utility only through ã (no direct effect of old prices on current utility conditional on current prices).

2. **State-level cigarette excise tax rates (τ_{s,t-L}).** Tax changes are plausibly exogenous to individual taste shocks and shift past cigarette prices → past consumption → current ã. Data available from CDC State Tobacco Activities Tracking and Evaluation (STATE) System or Tax Burden on Tobacco. Exclusion restriction: past tax rates affect current utility only through accumulated addiction.

3. **Initial household demographics as predictors of ã₀.** Household characteristics at panel entry (household size, income quartile, region, age of household head) predict initial addiction but are predetermined w.r.t. current taste shocks. These are valid instruments under the assumption that demographics affect current utility only through the channels already in the model (TYA indicator, prices).

4. **Hausman-type instruments: leave-out mean prices.** Average price of cigarettes/e-cigs in the household's state excluding the household's own purchases. Cross-sectional price variation driven by supply-side factors (distribution costs, state regulations) rather than local demand shocks. Standard in BLP-style demand estimation.

5. **Predicted addiction from first-stage regression.** Run a first-stage regression of ã on exogenous variables (lagged prices, taxes, demographics), then use fitted values ã̂ as an instrument. This is an IV/control function approach that avoids directly instrumenting in the GMM.

### Price Endogeneity

Prices may be endogenous if manufacturers/retailers set prices in response to local demand conditions (e.g., higher markups in areas with more addicted consumers). Instruments for prices should shift supply without directly affecting demand.

**Candidate instruments for prices (p_cig, p_ecig):**

1. **State excise tax rates (τ_{s,t}).** The most standard instrument for cigarette prices. Tax rates are set by legislatures and shift retail prices mechanically. E-cigarette taxes are newer and less common but increasingly available (as of 2023, ~30 states tax e-cigs). Exclusion restriction: taxes affect utility only through prices (reasonable if tax salience is low and consumers respond to shelf prices).

2. **Wholesale/manufacturer prices or input costs.** Tobacco leaf prices (USDA data), nicotine/PG/VG input costs for e-liquids, or wholesale price indices. These are supply-side cost shifters. Exclusion restriction: input costs affect consumer utility only through retail prices.

3. **Hausman instruments: prices in other markets.** Average price of the same product category in other states or regions. The identifying assumption is that cross-market price correlation is driven by common cost shocks (manufacturer pricing, input costs) rather than correlated demand shocks. Standard in differentiated products demand (Hausman 1996, Nevo 2001).

4. **Distance to distribution centers or retailer density.** Geographic supply-side variation that affects retail markups and distribution costs. Retailer density (convenience stores, vape shops per capita) affects local competition and pricing. Exclusion restriction: distribution infrastructure affects utility only through prices.

5. **Regulatory events as instruments (FDA PMTA decisions, state flavor ban announcements).** Timing of FDA marketing denial orders or state-level regulatory changes creates exogenous supply shifts for affected product categories. These are particularly useful for e-cigarette price variation. Must be careful about direct demand effects of regulatory salience (consumers may change preferences upon learning about bans, not just through prices).

6. **BLP-style product characteristics instruments.** Number of competing products in the market, characteristics of rival products. In this context: number of e-cigarette brands available in the household's market, or the variety of nicotine concentrations available. These shift the competitive environment and hence prices. Standard in IO demand estimation.

### Practical Recommendation

Start with the simplest credible specification:
- **For ã:** Lagged prices (L=2,3) + initial demographics. These are available in the existing data and require no additional data collection.
- **For prices:** State excise tax rates (readily available) + Hausman instruments (leave-out state means, computable from the HMS panel).

The first-stage F-statistics should be checked for weak instruments. If lagged prices are weak instruments for ã (possible if price variation is small relative to taste persistence), the tax + Hausman instruments for prices may carry more identification power.

## Implementation Plan

### Code Changes in `01_Functions.jl`

**New functions needed:**

```julia
function gmm_moments(
    V_choice,           # From VFI (same as MLE)
    N_J, N_P, A, P,
    y,                  # Observed choices
    tya_state,          # TYA state (1-4)
    a_continuous,       # Continuous addiction
    p_continuous,       # Continuous prices (N × 2)
    Z                   # Instrument matrix (N × L)
)
    # 1. Interpolate V_choice at each observation's continuous state
    #    (reuse trilinear interpolation logic from log_likelihood)
    # 2. Compute choice probabilities P(j | x_i; θ) via softmax
    # 3. Compute category-level residuals e_ik = 𝟙[y_i ∈ k] - Σ_{j∈k} P(j|x_i;θ)
    # 4. Form moments: ḡ = (1/N) Σ_i e_i ⊗ z_i
    # Returns: moment vector ḡ (K_cat × L vector)
end

function gmm_objective(
    ḡ,                  # Moment vector
    W                   # Weighting matrix
)
    # Returns: ḡ' W ḡ (scalar)
end

function gmm_standard_errors(
    θ_hat,              # Estimated parameters
    W,                  # Weighting matrix
    ... )
    # 1. Compute G (Jacobian of moments w.r.t. θ) via finite differences
    # 2. Compute Ω̂ (variance of moments) from observation-level moments
    # 3. Return (G'WG)⁻¹ G'W Ω̂ WG (G'WG)⁻¹
end
```

**Modified functions:**

```julia
function objective_gmm(θ_vec)
    # Same as objective() through VFI solve
    # Replace log_likelihood() call with:
    #   1. gmm_moments(V_decision, ..., Z)
    #   2. gmm_objective(ḡ, W)
    # Return scalar Q(θ)
end
```

### New files needed:
- `02_Second_Stage_Estimation/04_GMM_Estimation.jl` — Main GMM estimation script
  (mirrors `02_Estimation.jl` but uses `objective_gmm`)
- `02_Second_Stage_Estimation/05_GMM_Standard_Errors.jl` — GMM sandwich SEs

### Estimation Procedure

**One-step GMM (identity weighting):**
1. Set W = I (identity matrix)
2. Minimize Q(θ) = ḡ(θ)' ḡ(θ)
3. Consistent but inefficient

**Two-step GMM (optimal weighting):**
1. Estimate θ̂₁ with W = I
2. Compute Ω̂ = (1/N) Σᵢ g(yᵢ, xᵢ; θ̂₁) g(yᵢ, xᵢ; θ̂₁)' from first-step residuals
3. Set W = Ω̂⁻¹
4. Re-estimate θ̂₂ minimizing ḡ(θ)' Ω̂⁻¹ ḡ(θ)
5. This is efficient GMM (achieves semiparametric efficiency bound for these moments)

**Continuously-updated GMM (CUE):**
- W(θ) = Ω̂(θ)⁻¹ updated at every θ evaluation
- More robust to misspecification of the weighting matrix
- Computationally more expensive (recompute Ω̂ each eval)
- Probably not worth it given VFI is already the bottleneck

### Specification Tests

**Hansen's J-test:** Under H₀ (model correctly specified), N · Q(θ̂) ~ χ²(L - K) where L = number of moments, K = number of parameters. With 40 moments and 13 parameters: χ²(27). Rejection suggests the model's moment conditions are violated — either the utility specification is wrong or the instruments are invalid.

**Weak instrument diagnostics:** First-stage F-statistics for each endogenous variable. Staiger-Stock rule of thumb: F > 10. Particularly important for ã instruments since addiction is a constructed variable.

## Scope Assessment

Estimated effort: 2-3 days of coding, assuming existing VFI infrastructure is reused as-is.

| Component | New Code | Reused from MLE |
|-----------|----------|-----------------|
| VFI | — | 100% reused |
| Interpolation | — | 100% reused |
| Choice probabilities | ~20 lines | Softmax logic reused |
| Moment construction | ~80 lines | — |
| GMM objective | ~15 lines | — |
| GMM standard errors | ~100 lines | Finite diff logic adapted |
| Instrument matrix Z | ~50 lines | Data loading reused |
| Estimation script | ~150 lines | Structure from 02_Estimation.jl |
| Weighting matrix | ~30 lines | — |

Total new code: ~450 lines. The hardest part is constructing the instrument matrix Z, particularly if external data (tax rates) needs to be merged.

---

# GSD Single-Shock Error Structure Notes

Notes from reading Gowrisankaran & Schmidt-Dengler (2025), "A Computable Dynamic Oligopoly Model of Capacity Investment." Advisor suggested this error structure may be relevant for the ordered quantity bins in the dynamic model.

## The Issue with Multinomial Logit + Ordered Choices

The current model uses 31 i.i.d. Type I EV shocks (one per alternative). Within each product category, the quantity bins are ordered:
- 12 cigarette bins: 1, 2, 3-4, 5-9, 10, ..., 41+ packs
- 7 original ecig bins: 0-5, 5-10, ..., 50+ mL
- 7 flavored ecig bins: same
- 4 bundles

GSD (2025) shows that with many ordered choices and i.i.d. multinomial shocks:
- Investment/consumption levels do not converge as the grid refines
- Mean investment keeps increasing with more grid points (Table 1: 258 MW at 5 choices -> 384 MW at 80 choices)
- Ex-ante values grow without bound (diverge to infinity)
- The model increasingly resembles picking a random option (whichever has the highest i.i.d. draw)
- The Ackerberg-Rysman (2006) correction also fails to stabilize values

With a single linear shock to marginal cost (the GSD model):
- Results converge by ~10 grid points (mean investment stable at ~265 MW)
- Ex-ante values converge to ~$3,325M
- Some actions are never chosen (not on the discrete concave hull)

## The GSD Alternative: Single Shock + Cutoffs

Instead of K independent shocks, one scalar shock epsilon determines the ordered choice via threshold cutoffs.

**Period payoff structure:**
```
pi(a, s) - c_tilde(a, s) * epsilon
```
Where pi is the deterministic payoff, c_tilde is the stochastic cost component (linear in the action), and epsilon is a single scalar shock. The optimal action a(epsilon) is weakly decreasing in epsilon (higher cost shock -> less investment/consumption).

**Key results:**
- Proposition 1: Action k is chosen with positive probability iff it lies on the discrete concave hull of the choice-specific value function
- Proposition 2: An O(K) algorithm to find C(A) (the concave hull) and compute choice probabilities
- Corollary 3: At most 2K - 3 cutoff comparisons (linear), vs K(K-1)/2 for the naive approach

**Choice probability for action k:**
```
Pr(a = k) = Pr(epsilon_bar(k, k+1) <= epsilon < epsilon_bar(k-1, k))
           = F(epsilon_bar(k-1, k)) - F(epsilon_bar(k, k+1))
```
Where epsilon_bar(j, k) = (v^k - v^j) / (c_tilde(alpha^k) - c_tilde(alpha^j)) is the cutoff where the agent is indifferent between actions j and k.

## How This Would Apply to the Current Model

The idea is to restructure the 31-alternative choice as a two-level decision:

**Level 1 - Category choice (logit, ~5 options):**
- Outside option, cigarettes, original ecig, flavored ecig, bundles
- Small enough that multinomial logit is appropriate (GSD critique is about many ordered choices)

**Level 2 - Within-category quantity (single shock + cutoffs):**
- Given a category, household draws one scalar shock epsilon
- The shock determines the quantity bin via cutoffs on the GSD concave hull
- Some bins may have zero probability if not on the concave hull

**New parameters needed:** ~3-4 sigma parameters (scale of within-category shock), one per category.

## Implementation Scope

This is a major structural change, not a minor edit. Affected components:

1. **VFI (solve_vfi):** Rewrite. Instead of logsumexp over 31 alternatives, integrate within-category shock analytically using normal CDF at cutoffs, then logsumexp over ~5 categories.

2. **Likelihood (log_likelihood):** Rewrite. Choice probability becomes: Pr(category) * Pr(quantity bin | category). Within-category probability comes from shock CDF at cutoffs.

3. **New algorithm:** Implement discrete concave hull (Proposition 2) to find which bins within each category are chosen with positive probability.

4. **MC simulation (simulate_data):** Rewrite. Draw category from logit, then draw epsilon and apply cutoffs to determine quantity bin.

5. **Counterfactual and model validation:** Rewrite probability calculations.

6. **Standard errors:** Now 14-15 parameters instead of 11.

Estimated scope: ~500-800 lines of new/modified Julia code across all scripts.

## Mitigating Evidence

The static nested logit (04_Static_Nested_Logit.jl) estimated sigma_hat = 0.12 (t = 7.29) -- statistically significant but economically modest. Within-nest substitution is only ~14% stronger than across-nest. This suggests the IIA violation from using standard logit may not be severe in the current application, though the nested logit test is not identical to the GSD critique (nested logit still uses i.i.d. shocks within nests).

## Key References

- Gowrisankaran & Schmidt-Dengler (2025): The main paper. Python code at github.com/patohdzs/gsd-capacity-investment
- Kalouptsidi (2018): Shipbuilding application. Assumed all choices on concave hull (convexity of cost sufficient).
- Caoui (2023): Digital movie screens. Assumed "decreasing differences" for monotonicity.
- Gowrisankaran, Langer, & Reguant (2024): Energy transitions. Used GSD methods. Found some choices NOT on concave hull.
- Ackerberg & Rysman (2006): Proposed log(K) correction for multinomial logit with many options. GSD simulation shows this doesn't stabilize either.

## Decision

TBD -- discuss with advisor whether this is for the current paper or a future extension.
