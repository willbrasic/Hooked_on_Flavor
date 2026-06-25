################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script obtains the AR(1) parameters used for state-transitions
# in the holdout estimation. Estimated on ALL households x 2021-2022
# monthly median prices only (no 2023 data).
#
# AR parameters are written to AR_Parameters_Holdout/ so that
# 04_Price_State_Transitions_Holdout.R reads the correct parameters when
# generating holdout Halton draw transitions.
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

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt_full <- fread(file_name)

# Filter to ALL households x 2021-2022
dt_full[, year_pm := year(purchase_month)]
dt <- dt_full[year_pm %in% c(2021, 2022)]
dt[, year_pm := NULL]

cat("Holdout AR estimation computed from", nrow(dt), "observations (all HHs x 2021-2022)\n")
cat("Months in holdout data:", format(sort(unique(dt[, purchase_month])), "%Y-%m"), "\n")

# Create AR_Parameters_Holdout/ directory if it does not exist
dir_ar <- "../Dynamic_Model/AR_Parameters_Holdout/"
if (!dir.exists(dir_ar))
{
  dir.create(dir_ar)
}


#############################
# Prepare data
#############################

# Median per-pack price for cigarette purchases
median_cig_prices <- dt[cig == 1 | cig_ecig == 1,
                        .(median_real_per_pack_price = median(real_per_pack_price_paid, na.rm = TRUE)),
                        by = purchase_month][order(purchase_month)]

# Median per-mL price for e-cig purchases
median_ecig_prices <- dt[ecig == 1 | cig_ecig == 1,
                         .(median_real_per_mL_price = median(real_per_mL_price_paid, na.rm = TRUE)),
                         by = purchase_month][order(purchase_month)]

# Create single object with all median per-unit prices by month
dt_median_prices <- merge(
  median_cig_prices,
  median_ecig_prices,
  by = "purchase_month",
  all = TRUE
)


#############################
# AR1 estimation
#############################

# Cigarettes: AR(1) on median per-pack price
cig_ar <- median_cig_prices[
  , arima(median_real_per_pack_price, order = c(1, 0, 0))
]

# E-cigs: AR(1) on median per-mL price
ecig_ar <- median_ecig_prices[
  , arima(median_real_per_mL_price, order = c(1, 0, 0))
]

# Extract AR parameters
# R's arima() reports the process mean mu as "intercept", not the
# regression intercept. The regression form is p_{t+1} = phi_0 + phi_1 * p_t + eta,
# where phi_0 = mu * (1 - phi_1)
cig_params  <- coef(cig_ar)
ecig_params <- coef(ecig_ar)

# Combine into a data table with the correct regression intercept
ar_params <- data.table(
  series    = c("cig", "ecig"),
  intercept = c(cig_params["intercept"]  * (1 - cig_params["ar1"]),
                ecig_params["intercept"] * (1 - ecig_params["ar1"])),
  ar1       = c(cig_params["ar1"],
                ecig_params["ar1"])
)
ar_params


#############################
# Standard Error for
# AR1 phi_0 for cigs
#############################

# Gradient of phi_0 = mu(1 - phi_1) with respect to (rho, mu)
vcov_cig <- vcov(cig_ar)
mu_cig   <- cig_params["intercept"]
rho_cig  <- cig_params["ar1"]
grad_cig <- c(ar1 = unname(-mu_cig), intercept = unname(1 - rho_cig))

# Delta method variance
var_phi0_cig <- t(grad_cig) %*% vcov_cig %*% grad_cig
se_phi0_cig  <- sqrt(var_phi0_cig)
se_phi0_cig


#############################
# Standard Error for
# AR1 phi_0 for ecigs
#############################

# Gradient of phi_0 = mu(1 - phi_1) with respect to (rho, mu)
vcov_ecig <- vcov(ecig_ar)
mu_ecig   <- ecig_params["intercept"]
rho_ecig  <- ecig_params["ar1"]
grad_ecig <- c(ar1 = unname(-mu_ecig), intercept = unname(1 - rho_ecig))

# Delta method variance
var_phi0_ecig <- t(grad_ecig) %*% vcov_ecig %*% grad_ecig
se_phi0_ecig  <- sqrt(var_phi0_ecig)
se_phi0_ecig


#############################
# Estimate covariance
# matrix
#############################

# Get AR residuals for each series
resid_cig  <- residuals(cig_ar)
resid_ecig <- residuals(ecig_ar)

# Stack residuals into a matrix
resid_matrix <- cbind(
  cig  = resid_cig,
  ecig = resid_ecig
)

# Variance-covariance matrix of shocks
Sigma_hat <- cov(resid_matrix, use = "complete.obs")


#############################
# Write results to a file
#############################

# Write median per-unit monthly prices to Data_Holdout/
file_name_medians <- "../Dynamic_Model/Data_Holdout/Median_Per-Unit_Monthly_Prices.csv"
fwrite(dt_median_prices, file_name_medians)

if (file.exists(file_name_medians))
{
  cat("Results have been written to\n", file_name_medians, "\n")
} else
{
  cat("Error: File could not be written\n")
}

# Write AR parameters to AR_Parameters_Holdout/
file_name_ar <- paste0(dir_ar, "AR_Parameters_Phi.csv")
fwrite(ar_params, file_name_ar)

if (file.exists(file_name_ar))
{
  cat("Results have been written to\n", file_name_ar, "\n")
} else
{
  cat("Error: File could not be written\n")
}

# Convert Sigma_hat to data table and write
dt_Sigma_hat <- as.data.table(Sigma_hat)
dt_Sigma_hat[, series := rownames(Sigma_hat)]
setcolorder(dt_Sigma_hat, c("series", setdiff(colnames(dt_Sigma_hat), "series")))

file_name_sigma <- paste0(dir_ar, "AR_Parameters_Sigma.csv")
fwrite(dt_Sigma_hat, file_name_sigma)

if (file.exists(file_name_sigma))
{
  cat("Results have been written to\n", file_name_sigma, "\n")
} else
{
  cat("Error: File could not be written\n")
}
