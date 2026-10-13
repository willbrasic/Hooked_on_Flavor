################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script creates the pricing spaces (the price grid of the dynamic
# model) from the market prices written by 01_Data_Prep_Holdout.R, and reports the
# within-household price variation those market prices carry.
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

# Folder holding the model inputs (all households x 2021-2022, the holdout estimation sample)
dir_data <- "../Dynamic_Model/Data_Holdout/"


#############################
# Pricing spaces by category
#############################

# The price state is the market price each household faces (county-month
# median when the county cell is large enough, else state-month, else
# national; see 01_Data_Prep.R), so the grid is built from the distribution
# of those market prices across household-months rather than from
# transaction prices, which are more dispersed and would place grid points
# where no price state lies. 
file_name <- paste0(dir_data, "Prices.csv")
dt_market <- fread(file_name, select = c("market_cig", "market_ecig"))

# 7-point quantile grid. The end points sit at the 2.5th and 97.5th
# percentiles so that 5% of household-months are clamped to the grid edge
price_quantiles <- c(0.025, 0.20, 0.35, 0.50, 0.65, 0.80, 0.975)

# Cigarette and e-cigarette market price quantiles
cig_prices  <- dt_market[, as.list(quantile(market_cig,  probs = price_quantiles))]
ecig_prices <- dt_market[, as.list(quantile(market_ecig, probs = price_quantiles))]

# Add labels
cig_prices[, type := "cig"]
ecig_prices[, type := "ecig"]

# Combine all into one data table
dt_prices <- rbindlist(list(cig_prices, ecig_prices), use.names = TRUE, fill = TRUE)

# Move `type` to the front
setcolorder(dt_prices, c("type", setdiff(names(dt_prices), "type")))

# Convert to long format
dt_prices_long <- melt(
  dt_prices,
  id.vars = "type",
  variable.name = "percentile",
  value.name = "price"
)
dt_prices_long <- dcast(
  dt_prices_long,
  percentile ~ type,
  value.var = "price"
)
dt_prices_long

# Write to file
file_name <- paste0(dir_data, "Pricing_Spaces.csv")
fwrite(dt_prices_long, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Within-Household Price Variation
# (Price Identification Diagnostics)
#############################

# Household and month for each row of Prices.csv, in the same row order
dt_hh_prices <- fread(paste0(dir_data, "TYA_States.csv"), select = c("household_code", "purchase_month"))
dt_hh_prices[, `:=` (market_cig = dt_market[, market_cig], market_ecig = dt_market[, market_ecig])]

# Map each market price to the nearest grid point (1 = lowest), matching how
# the Julia code assigns the observed price state
cig_grid  <- dt_prices_long[, cig]
ecig_grid <- dt_prices_long[, ecig]
nearest_state <- function(price, grid)
{
  apply(abs(outer(price, grid, "-")), 1, which.min)
}
dt_hh_prices[, cig_price_state  := nearest_state(market_cig,  cig_grid)]
dt_hh_prices[, ecig_price_state := nearest_state(market_ecig, ecig_grid)]

# Share of household-months nearest each grid point
cat("\nShare of household-months at each grid point (cig / ecig):\n")
print(rbind(cig  = round(prop.table(table(factor(dt_hh_prices[, cig_price_state],  levels = seq_along(cig_grid)))),  3),
            ecig = round(prop.table(table(factor(dt_hh_prices[, ecig_price_state], levels = seq_along(ecig_grid)))), 3)))

# Within-household price variation: SD and price state transitions
dt_cig_var <- dt_hh_prices[, .(
  mean_cig_price  = mean(market_cig),
  sd_cig_price    = sd(market_cig),
  n_months        = .N,
  n_cig_states    = uniqueN(cig_price_state),
  n_state_changes = sum(diff(cig_price_state) != 0)
), keyby = household_code]

dt_ecig_var <- dt_hh_prices[, .(
  mean_ecig_price = mean(market_ecig),
  sd_ecig_price   = sd(market_ecig),
  n_months        = .N,
  n_ecig_states   = uniqueN(ecig_price_state),
  n_state_changes = sum(diff(ecig_price_state) != 0)
), keyby = household_code]

# Summary statistics
cat("\n--- Cigarette Price Variation (within household, market price) ---\n")
print(dt_cig_var[, .(
  mean_sd            = mean(sd_cig_price,     na.rm = TRUE),
  median_sd          = median(sd_cig_price,   na.rm = TRUE),
  pct_0_transitions  = mean(n_state_changes == 0) * 100,
  mean_transitions   = mean(n_state_changes),
  median_transitions = median(n_state_changes),
  mean_n_states      = mean(n_cig_states)
)])

cat("\n--- E-Cigarette Price Variation (within household, market price) ---\n")
print(dt_ecig_var[, .(
  mean_sd            = mean(sd_ecig_price,    na.rm = TRUE),
  median_sd          = median(sd_ecig_price,  na.rm = TRUE),
  pct_0_transitions  = mean(n_state_changes == 0) * 100,
  mean_transitions   = mean(n_state_changes),
  median_transitions = median(n_state_changes),
  mean_n_states      = mean(n_ecig_states)
)])

# Distribution of price state transitions per household
cat("\n--- Cig price state transition distribution ---\n")
print(dt_cig_var[, .N, keyby = n_state_changes])

cat("\n--- E-cig price state transition distribution ---\n")
print(dt_ecig_var[, .N, keyby = n_state_changes])
