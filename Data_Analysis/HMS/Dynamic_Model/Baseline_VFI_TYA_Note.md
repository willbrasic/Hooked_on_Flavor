# Baseline Model: TYA Treatment in VFI vs Likelihood

In the baseline model (`02_Second_Stage_Estimation/01_Functions.jl`),
TYA is a binary state: 1 = no TYA in household, 2 = TYA present.

## VFI (`solve_vfi` / `solve_vfi_sophisticated`)

```
V_choice[1, j, a, p] = U[1, j, a, p] + δ * EV_1
V_choice[2, j, a, p] = U[2, j, a, p] + δ * EV_2
```

Each TYA state's continuation value (`EV_1`, `EV_2`) only references
its own future value — a household in TYA state 1 today expects to
be in TYA state 1 tomorrow, and likewise for state 2. There is no
TYA transition matrix. Consumers do not anticipate changes in their
household's TYA composition when making forward-looking decisions.

## Likelihood (`log_likelihood`)

```
tya_idx = tya_state[i]
```

The likelihood evaluates each observation at the household's actual
TYA state in that month. Since `tya_state` is observation-specific
(household x month), it varies within a household over the panel.
A household that starts without a teen and later has a child enter
the 13-25 age range will have `tya_state = 1` in early months and
`tya_state = 2` in later months. The data captures these transitions.

## Summary

- **VFI**: TYA is treated as absorbing (no transitions).
  Consumers solve as if their current TYA state is permanent.
- **Likelihood**: TYA is allowed to change over time within a household.
  Each observation is evaluated at whatever TYA state the
  household actually has in that month.

The model therefore does not capture forward-looking behavior with
respect to TYA changes — a household approaching the TYA threshold
(e.g., oldest child is 12) does not anticipate the shift in flavor
utility that will occur when the child turns 13. The value function
is solved under the assumption that today's TYA state persists
forever, but the likelihood correctly conditions on the realized
TYA path.

## Beta Variant

The beta variant (`01_Functions_Beta.jl`) relaxes this by expanding
TYA to 4 states (no-TYA stable, no-TYA approaching, TYA present
stable, TYA ending soon) and introducing a transition matrix `Π_tya`
so that consumers integrate over possible future TYA states:

```
V_choice[tya, j, a, p] = U[tya, j, a, p]
                        + δ * Σ_{tya'} Π[tya, tya'] * EV_{tya'}
```
