################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script creates the pricing spaces for holdout estimation.
# Computed from ALL households x 2021-2022 to match the holdout
# estimation sample defined in 01_Data_Prep_Holdout.R.
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

cat("Holdout pricing spaces computed from", nrow(dt), "observations (all HHs x 2021-2022)\n")


#############################
# Pricing spaces by category
#############################

# 7-point quantile grid (5th, 20th, 35th, 50th, 65th, 80th, 95th percentiles)
price_quantiles <- c(0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95)

# Cigarette price quantiles
cig_prices <- dt[cig == 1 | cig_ecig == 1, as.list(quantile(real_per_pack_price_paid, probs = price_quantiles, na.rm = TRUE))]

# E-cig price quantiles
ecig_prices <- dt[ecig == 1 | cig_ecig == 1, as.list(quantile(real_per_mL_price_paid, probs = price_quantiles, na.rm = TRUE))]

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
file_name <- "../Dynamic_Model/Data_Holdout/Pricing_Spaces.csv"
fwrite(dt_prices_long, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}
