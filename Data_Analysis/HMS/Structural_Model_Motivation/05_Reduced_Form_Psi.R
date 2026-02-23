################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script estimates psi (addiction decay rate) from the reduced-form
# persistence of nicotine consumption using an AR(1) specification. The
# estimate is used as a first-stage input to the dynamic structural model
# to break the psi-mu identification problem revealed by MC simulations.
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
                "4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data/")
setwd(wd)

# Load packages
pacman::p_load(data.table, fixest)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 20)


#############################
# Load data
#############################

# Load in panel data
dt_hh  <- fread(paste0("./Household_Codes.csv"))
dt_tya <- fread(paste0("./Teen_Young_Adult.csv"))

# Product choices 
dt_choices <- fread(paste0("./Product_Choices.csv"))

# Nicotine and consumption per alternative
dt_nicotine <- fread(paste0("./Nicotine_Spaces.csv"))
dt_consumption <- fread(paste0("./Consumption_Spaces.csv"))

# Prices 
dt_prices <- fread(paste0("./Prices.csv"))


#############################
# Construct nicotine and
# consumption vectors
#############################

# Product choice column names 
alt_names <- names(dt_choices)

# Build nicotine vector aligned to the product choice columns
# Non-bundle alternatives map directly; bundles sum cig + ecig nicotine components
nicotine_vec <- numeric(length(alt_names))
names(nicotine_vec) <- alt_names

for (i in seq_along(alt_names))
{
  alt <- alt_names[i]

  if (alt == "outside_option")
  {
    nicotine_vec[i] <- 0
  } else if (grepl("^bundle_", alt))
  {
    # Bundle alternatives: sum cig and ecig nicotine components
    # e.g., bundle_orig_lo -> bundle_orig_lo_cig_nic + bundle_orig_lo_ecig_nic
    cig_name  <- paste0(alt, "_cig_nic")
    ecig_name <- paste0(alt, "_ecig_nic")
    nicotine_vec[i] <- dt_nicotine[alternative == cig_name, nicotine] +
                       dt_nicotine[alternative == ecig_name, nicotine]
  } else
  {
    # Cig and ecig alternatives map directly
    nicotine_vec[i] <- dt_nicotine[alternative == alt, nicotine]
  }
}

# Build cig and ecig consumption vectors (separate by product type)
cig_cons_vec  <- numeric(length(alt_names))
ecig_cons_vec <- numeric(length(alt_names))
names(cig_cons_vec)  <- alt_names
names(ecig_cons_vec) <- alt_names

for (i in seq_along(alt_names))
{
  alt <- alt_names[i]

  if (alt == "outside_option")
  {
    cig_cons_vec[i]  <- 0
    ecig_cons_vec[i] <- 0
  } else if (grepl("^cig_", alt))
  {
    # Pure cig alternatives
    cig_cons_vec[i]  <- dt_consumption[alternative == alt, consumption]
    ecig_cons_vec[i] <- 0
  } else if (grepl("^orig_ecig_|^flav_ecig_", alt))
  {
    # Pure ecig alternatives
    cig_cons_vec[i]  <- 0
    ecig_cons_vec[i] <- dt_consumption[alternative == alt, consumption]
  } else if (grepl("^bundle_", alt))
  {
    # Bundle alternatives: separate cig and ecig components
    # e.g., bundle_orig_lo -> bundle_orig_lo_cig + bundle_orig_lo_ecig
    cig_name  <- paste0(alt, "_cig")
    ecig_name <- paste0(alt, "_ecig")
    cig_cons_vec[i]  <- dt_consumption[alternative == cig_name, consumption]
    ecig_cons_vec[i] <- dt_consumption[alternative == ecig_name, consumption]
  }
}


#############################
# Compute household-month
# nicotine and consumption
#############################

# Convert choices to matrix for vectorized multiplication
mat_choices <- as.matrix(dt_choices)

# Nicotine intake: n_it = sum_j(choice_j * nicotine_j)
nicotine_it <- as.numeric(mat_choices %*% nicotine_vec)

# Cig consumption: sum_j(choice_j * cig_consumption_j)
cig_consumption_it <- as.numeric(mat_choices %*% cig_cons_vec)

# Ecig consumption: sum_j(choice_j * ecig_consumption_j)
ecig_consumption_it <- as.numeric(mat_choices %*% ecig_cons_vec)


#############################
# Compute aggregate prices
#############################

# Cig price columns 
cig_price_cols <- grep("^cig_", names(dt_prices), value = TRUE)

# Ecig price columns 
ecig_price_cols <- grep("^ecig_", names(dt_prices), value = TRUE)

# Mean cig and ecig price per observation 
mean_cig_price  <- rowMeans(as.matrix(dt_prices[, ..cig_price_cols]))
mean_ecig_price <- rowMeans(as.matrix(dt_prices[, ..ecig_price_cols]))


#############################
# Merge into panel dataset
#############################

# Combine all variables into a single data.table
dt <- data.table(
  household_code   = dt_hh[, household_code],
  purchase_month   = dt_tya[, purchase_month],
  nicotine         = nicotine_it,
  cig_consumption  = cig_consumption_it,
  ecig_consumption = ecig_consumption_it,
  mean_cig_price   = mean_cig_price,
  mean_ecig_price  = mean_ecig_price,
  tya              = dt_tya[, teen_or_young_adult_present]
)


#############################
# Create lagged variables
#############################

# Lagged nicotine, cig consumption, and ecig consumption within household
dt[, `:=` (
  lag_nicotine         = shift(nicotine, n = 1),
  lag_cig_consumption  = shift(cig_consumption, n = 1),
  lag_ecig_consumption = shift(ecig_consumption, n = 1)
), by = household_code]

# Drop first observation per household (no lag available)
dt <- dt[!is.na(lag_nicotine)]


#############################
# Nicotine AR(1)
#############################

# Regression of nicotine on lagged nicotine with household and month FEs
model_1 <- feols(
  nicotine ~ lag_nicotine | household_code + purchase_month,
  data    = dt,
  cluster = ~household_code
)


#############################
# Nickell Bias Correction
#############################

# With T = 36 and large N, the FE estimator is nearly consistent. The Nickell
# (1981) bias is O(1/T): bias ≈ -(1 + ρ) / (T - 1), which is small for small  T 
# Solving for the corrected ρ:
#   ρ_corrected = (ρ_FE * (T_bar - 1) + 1) / (T_bar - 2)

# Extract estimates
rho_fe <- coef(model_1)["lag_nicotine"]
se_fe  <- se(model_1)["lag_nicotine"]

# Average number of time periods per household (for unbalanced panel correction)
T_bar <- dt[, .N, by = household_code][, mean(N)]

# Bias-corrected rho
rho_corrected <- (rho_fe * (T_bar - 1) + 1) / (T_bar - 2)

# Extract psi
psi_corrected <- 1 - rho_corrected
psi_corrected

# SE is approximately unchanged (correction is a linear function of rho_hat)
# SE_corrected = SE_FE * (T_bar - 1) / (T_bar - 2)
se_corrected <- se_fe * (T_bar - 1) / (T_bar - 2)
se_corrected




