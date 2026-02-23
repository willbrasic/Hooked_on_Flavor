# Sophisticated vs Naive VFI for Quasi-Hyperbolic Discounting

This note explains how value function iteration differs between
naive and sophisticated agents under quasi-hyperbolic (beta-delta)
discounting, and why all code now uses `solve_vfi_sophisticated`.

## Setup

Under quasi-hyperbolic discounting, the agent discounts the immediate
future by beta * delta and all subsequent periods by delta:

```
U_0 + βδ U_1 + βδ² U_2 + βδ³ U_3 + ...
```

When beta = 1, this collapses to standard exponential discounting.
When beta < 1, the agent is present-biased — they disproportionately
favor immediate payoffs over all future payoffs.

## Naive Agent (`solve_vfi`)

A naive agent has present bias but **does not realize their future
self will also be present-biased**. They think: "I'm impatient right
now, but starting tomorrow I'll discount exponentially with delta."

Since the agent believes their future self is a patient delta-discounter,
VFI solves the Bellman equation with delta only:

```
V_choice[j] = U[j] + δ · EV
V[a, p]     = logsumexp(V_choice)
```

The continuation value `V` reflects the agent's (incorrect) belief
that their future self will maximize `V_choice`. After convergence,
beta is applied post-hoc to get decision utility — what the agent
actually uses to choose today:

```
V_decision[j] = U[j] + β·δ · EV
              = (1 - β) · U[j] + β · V_choice[j]
```

The post-hoc transformation works because `δ·EV = V_choice - U`,
so `U + βδ·EV = U + β·(V_choice - U) = (1-β)·U + β·V_choice`.

## Sophisticated Agent (`solve_vfi_sophisticated`)

A sophisticated agent **knows** their future self will also be
present-biased. They correctly predict: "Tomorrow, my future self
will over-weight immediate utility and under-weight the continuation
— just like I do today."

This changes the continuation value. The future self will choose
according to decision utility `V_d` (which includes present bias),
not experienced utility `W`. So we need two arrays each iteration:

```
V_d[j] = U[j] + β·δ · EV    (decision utility — what future self uses to CHOOSE)
W[j]   = U[j] + δ · EV      (experienced utility — what actually ACCRUES)
```

The future self's choice probabilities come from `V_d`:

```
p_j = softmax(V_d)_j = exp(V_d[j]) / Σ_k exp(V_d[k])
```

But the actual value that results from those choices is `W`, not `V_d`.
The sophisticated agent knows this. Under the Type I extreme value
(logit) error assumption, the expected value of choosing optimally is:

```
V[a, p] = Σ_j p_j · W[j] + H(p)
```

where `H(p) = -Σ_j p_j · log(p_j)` is the entropy from the T1EV
shocks. This entropy term captures the option value of randomness —
even though the agent makes mistakes (from the econometrician's
perspective), those "mistakes" have value because the agent observes
their private shock realizations.

In words: the sophisticated agent says "my future self will choose
with probabilities `p` (driven by `V_d`), the actual payoffs from
those choices are `W`, and the T1EV shocks contribute `H(p)` in
expectation."

## Why Sophisticated = Naive When beta = 1

When beta = 1:

1. `V_d = W` (both equal `U + δ·EV`)
2. `p_j = softmax(W)_j`
3. `Σ_j p_j · W_j + H(p) = logsumexp(W)`

Step 3 follows from a standard identity: under logit,
`logsumexp(v) = Σ_j softmax(v)_j · v_j + H(softmax(v))`.

So the sophisticated aggregation collapses to exactly what the
naive `solve_vfi` computes. All outputs are numerically identical.

## Key Economic Difference

The naive agent's continuation value is **too optimistic**. They
think their future self will be a patient delta-discounter who
maximizes `W = U + δ·EV`. In reality, the future self maximizes
`V_d = U + βδ·EV`, which over-weights immediate utility.

The sophisticated agent correctly accounts for this distortion.
Their continuation value is lower than the naive agent's because
they know future present bias will destroy some value — the future
self will sometimes choose high immediate utility at the expense
of long-run welfare.

```
                    Continuation V          Decision utility V_decision
Naive:              logsumexp(U + δ·EV)     (1-β)·U + β·V_choice  (post-hoc)
Sophisticated:      Σ softmax(V_d)·W + H    V_d = U + βδ·EV       (direct)
```

## Numerical Implementation

The aggregation `Σ_j p_j · W_j + H(p)` is computed in a numerically
stable way by factoring out the max of `V_d`:

```julia
vd_max = maximum(V_d[tya, :, a, p])
sum_exp = Σ_j exp(V_d[j] - vd_max)
log_denom = log(sum_exp)

agg = 0.0
for j in 1:N_J
    log_pj = V_d[j] - vd_max - log_denom
    pj = exp(log_pj)
    agg += pj * W[j] - pj * log_pj       # p_j · W_j + p_j · (-log p_j)
end
V_next[tya, a, p] = agg
```

This avoids overflow from large exponentials and is exact up to
floating-point precision.

## Files Changed

All estimation, MC simulation, model validation, and counterfactual
code now calls `solve_vfi_sophisticated` instead of `solve_vfi`.
The naive `solve_vfi` is retained in both `01_Functions.jl` and
`01_Functions_Beta.jl` for reference.

**Baseline (2 TYA states, `01_Functions.jl`):**
- `01_Functions.jl` — `objective()`
- `02_Model_Validation.jl`
- `02_Counterfactual_Flavor_Ban.jl` (status quo + ban)
- `01_MC_Simulation_Functions.jl` — `objective_mc()`
- `02_MC_Simulation_Array.jl` — DGP true VFI

**Beta variant (4 TYA states, `01_Functions_Beta.jl`):**
- `01_Functions_Beta.jl` — `objective()`
- `01_MC_Simulation_Functions_Beta.jl` — `objective_mc()`
- `02_MC_Simulation_Array_Beta.jl` — DGP true VFI
