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
