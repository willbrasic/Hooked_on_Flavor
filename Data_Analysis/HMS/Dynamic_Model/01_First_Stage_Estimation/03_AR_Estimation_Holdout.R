################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script estimates the AR(1) price process used for the price state
# transitions of the dynamic model for holdout estimation (all households x 2021-2022). The process is estimated on
# the panel of market prices households face (county-month, state-month, or
# national medians; see 01_Data_Prep_Holdout.R), instrumenting the lagged price with its
# second lag so that sampling noise in thin county and state cells does not
# attenuate the persistence estimate (Griliches and Hausman, 1986), and the
# innovation covariance is corrected for that noise. The instrument is valid
# when the noise is serially uncorrelated; the script checks this by also
# estimating with the third and fourth lags, which would give higher
# persistence than the second lag if the noise persisted. On the 2021-2023
# panel the three agree (cigarettes 0.989, 0.994, 0.992; e-cigarettes 0.969,
# 1.021, 0.994) and the autocovariance ratios at lags 2-4 are flat, so the
# noise is close to white. E-cigarette persistence is near a unit root and is
# capped at 0.99 so the price process is stationary on the grid.
################################################################################


#############################
# Preliminaries
#############################

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# Set working directory
wd <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/",
                "4th_Year_Paper_Data/HMS/2021-Onward/Tobacco_Panelists_Purchases_2021-Onward/")
setwd(wd)

# Load packages
pacman::p_load(data.table)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Folders holding the model inputs and the AR parameters
dir_data <- "../Dynamic_Model/Data_Holdout/"
dir_ar   <- "../Dynamic_Model/AR_Parameters_Holdout/"
if (!dir.exists(dir_ar))
{
  dir.create(dir_ar)
}


#############################
# Market price panel
#############################

# Market prices per household-month (row order of Prices.csv) and their keys
dt_market <- fread(paste0(dir_data, "Prices.csv"), select = c("market_cig", "market_ecig"))
dt_keys   <- fread(paste0(dir_data, "TYA_States.csv"), select = c("household_code", "purchase_month"))
stopifnot("Prices.csv and TYA_States.csv must have the same number of rows" = nrow(dt_market) == nrow(dt_keys))
dt_panel  <- cbind(dt_keys, dt_market)

# State of each household-month (for clustered standard errors)
dt_state <- fread("./all_panelists_purchases_monthly_CLEANED_2021-Onward.csv",
                  select = c("household_code", "purchase_month", "fips_state_code"))
dt_panel <- merge(dt_panel, dt_state, by = c("household_code", "purchase_month"), all.x = TRUE)
stopifnot("Every household-month needs a state" = !anyNA(dt_panel[, fips_state_code]))
setorder(dt_panel, household_code, purchase_month)

# First to fourth lags within household, only across consecutive months
dt_panel[, `:=` (
  lag1_cig  = shift(market_cig,  1), lag2_cig  = shift(market_cig,  2), lag3_cig  = shift(market_cig,  3), lag4_cig  = shift(market_cig,  4),
  lag1_ecig = shift(market_ecig, 1), lag2_ecig = shift(market_ecig, 2), lag3_ecig = shift(market_ecig, 3), lag4_ecig = shift(market_ecig, 4),
  month_gap2 = as.integer(round((as.numeric(purchase_month) - as.numeric(shift(purchase_month, 2))) / 30.44)),
  month_gap4 = as.integer(round((as.numeric(purchase_month) - as.numeric(shift(purchase_month, 4))) / 30.44))
), by = household_code]
dt_est <- dt_panel[month_gap2 == 2]   # estimation sample: two consecutive lags
dt_chk <- dt_panel[month_gap4 == 4]   # diagnostic sample: four consecutive lags
cat("Household-months used (two consecutive lags available):", nrow(dt_est), "of", nrow(dt_panel), "
")

# Maximum persistence written to the transitions (keeps the process stationary)
rho_cap <- 0.99


#############################
# AR(1) estimation
# (second lag as instrument)
#############################

# Two-stage least squares of p_t on p_{t-1} with p_{t-2} as the instrument.
# Under classical measurement error in the market price (thin cells), OLS on
# the lag is attenuated; the second lag is correlated with the true lagged
# price but not with the noise in the first lag, so the IV estimate of the
# persistence is consistent. Standard errors are clustered by state
estimate_ar_iv <- function(p, lag1, lag2, cluster)
{
  n   <- length(p)
  Z   <- cbind(1, lag2)                                   # instruments
  X   <- cbind(1, lag1)                                   # regressors
  PzX <- Z %*% solve(crossprod(Z), crossprod(Z, X))       # first-stage fitted X
  b   <- solve(crossprod(PzX, X), crossprod(PzX, p))      # 2SLS estimate
  u   <- as.vector(p - X %*% b)                           # structural residual

  # Cluster-robust sandwich
  A     <- solve(crossprod(PzX, X))
  score <- PzX * u
  meat  <- crossprod(rowsum(score, cluster))
  G     <- length(unique(cluster))
  V     <- (G / (G - 1)) * A %*% meat %*% t(A)

  list(intercept = b[1], ar1 = b[2], se_intercept = sqrt(V[1, 1]), se_ar1 = sqrt(V[2, 2]), resid = u, n = n, G = G)
}

cig_fit  <- with(dt_est, estimate_ar_iv(market_cig,  lag1_cig,  lag2_cig,  fips_state_code))
ecig_fit <- with(dt_est, estimate_ar_iv(market_ecig, lag1_ecig, lag2_ecig, fips_state_code))

# Cap the persistence (the intercept is re-derived so the long-run mean is unchanged)
for (nm in c("cig_fit", "ecig_fit"))
{
  f <- get(nm)
  if (f$ar1 > rho_cap)
  {
    cat("Capping", nm, "persistence from", round(f$ar1, 4), "to", rho_cap, "
")
    f$intercept <- f$intercept / (1 - f$ar1) * (1 - rho_cap)
    f$ar1       <- rho_cap
    assign(nm, f)
  }
}

# Instrument-depth diagnostic: with serially correlated noise, deeper lags give higher persistence
iv_rho <- function(p, lag1, z) cov(p, z) / cov(lag1, z)
cat("
Instrument-depth check (persistence by instrument lag 2 / 3 / 4; should agree):
")
cat("  cig :", paste(round(with(dt_chk, c(iv_rho(market_cig,  lag1_cig,  lag2_cig),  iv_rho(market_cig,  lag1_cig,  lag3_cig),  iv_rho(market_cig,  lag1_cig,  lag4_cig))),  3), collapse = " / "), "
")
cat("  ecig:", paste(round(with(dt_chk, c(iv_rho(market_ecig, lag1_ecig, lag2_ecig), iv_rho(market_ecig, lag1_ecig, lag3_ecig), iv_rho(market_ecig, lag1_ecig, lag4_ecig))), 3), collapse = " / "), "
")

# Innovation variance net of measurement noise. With p = p* + v (noise v
# independent over time and across categories), Cov(p_t, p_{t-1}) = rho Var(p*),
# so Var(p*) = Cov(p_t, p_{t-1}) / rho and Var(eta) = Var(p*) (1 - rho^2).
# The cross-category covariance of the structural residuals is unaffected by
# noise and is used directly
innovation_var <- function(p, lag1, rho)
{
  v_star <- cov(p, lag1) / rho
  v_star * (1 - rho^2)
}
var_eta_cig  <- with(dt_est, innovation_var(market_cig,  lag1_cig,  cig_fit$ar1))
var_eta_ecig <- with(dt_est, innovation_var(market_ecig, lag1_ecig, ecig_fit$ar1))
cov_eta      <- cov(cig_fit$resid, ecig_fit$resid)
Sigma_hat    <- matrix(c(var_eta_cig, cov_eta, cov_eta, var_eta_ecig), 2, 2, dimnames = list(c("cig", "ecig"), c("cig", "ecig")))
stopifnot("Innovation covariance must be positive definite" = all(eigen(Sigma_hat)$values > 0))

# AR parameters in the form used by the transitions script and the Julia code
ar_params <- data.table(
  series    = c("cig", "ecig"),
  intercept = c(cig_fit$intercept, ecig_fit$intercept),
  ar1       = c(cig_fit$ar1, ecig_fit$ar1)
)

# Report, with the alternatives for reference
ols_rho <- function(p, lag1) unname(coef(lm(p ~ lag1))[2])
cat("\nAR(1) price process (market prices, household-month panel, second-lag IV):\n")
print(data.table(
  series          = c("cig", "ecig"),
  intercept       = round(c(cig_fit$intercept, ecig_fit$intercept), 4),
  se_intercept    = round(c(cig_fit$se_intercept, ecig_fit$se_intercept), 4),
  ar1             = round(c(cig_fit$ar1, ecig_fit$ar1), 4),
  se_ar1          = round(c(cig_fit$se_ar1, ecig_fit$se_ar1), 4),
  long_run_mean   = round(c(cig_fit$intercept / (1 - cig_fit$ar1), ecig_fit$intercept / (1 - ecig_fit$ar1)), 3),
  sd_eta          = round(sqrt(c(var_eta_cig, var_eta_ecig)), 4),
  ols_ar1_for_ref = round(c(with(dt_est, ols_rho(market_cig, lag1_cig)), with(dt_est, ols_rho(market_ecig, lag1_ecig))), 4),
  n_obs           = c(cig_fit$n, ecig_fit$n),
  n_clusters      = c(cig_fit$G, ecig_fit$G)
))
cat("Innovation covariance matrix:\n"); print(round(Sigma_hat, 5))
cat("Innovation correlation:", round(cov_eta / sqrt(var_eta_cig * var_eta_ecig), 3), "\n")

#############################
# Write results to a file
#############################

# Write AR parameters to a file
file_name_ar <- paste0(dir_ar, "AR_Parameters_Phi.csv")
fwrite(ar_params, file_name_ar)

# Confirm results have been written to a file
if (file.exists(file_name_ar))
{
  cat("Results have been written to\n", file_name_ar, "\n")
} else
{
  cat("Error: File could not be written\n")
}

# Write standard errors to a file (for the appendix table)
dt_ar_se <- data.table(series = c("cig", "ecig"),
                       se_intercept = c(cig_fit$se_intercept, ecig_fit$se_intercept),
                       se_ar1 = c(cig_fit$se_ar1, ecig_fit$se_ar1),
                       n_obs = c(cig_fit$n, ecig_fit$n), n_clusters = c(cig_fit$G, ecig_fit$G))
fwrite(dt_ar_se, paste0(dir_ar, "AR_Parameters_SE.csv"))

# Convert Sigma_hat to data table
dt_Sigma_hat <- as.data.table(Sigma_hat)
dt_Sigma_hat[, series := rownames(Sigma_hat)]
setcolorder(dt_Sigma_hat, c("series", setdiff(colnames(dt_Sigma_hat), "series")))

# Write covariance matrix to file
file_name_sigma <- paste0(dir_ar, "AR_Parameters_Sigma.csv")
fwrite(dt_Sigma_hat, file_name_sigma)

# Confirm results have been written to a file
if (file.exists(file_name_sigma))
{
  cat("Results have been written to\n", file_name_sigma, "\n")
} else
{
  cat("Error: File could not be written\n")
}
