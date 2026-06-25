################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Relapse analysis tracking households who stop purchasing tobacco ("quit
# episodes") and measuring months until they restart. Under exponential
# discounting, a household that finds it optimal to stop should remain stopped
# absent a sufficiently large shock. Under present bias, the household may plan
# to stay quit but re-optimize when tomorrow becomes today, generating systematic
# relapse. High relapse rates, especially rapid ones, are consistent with beta < 1.
#
# Results are broken down by product type (cigarettes vs e-cigarettes).
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
options(datatable.print.nrows = 100)

# Load in full panel data
dt <- fread(paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv"))


#############################
# Helper function:
# compute relapse episodes
#############################

# Returns a list with two elements given a data.table and a purchase indicator:
#   (1) a data.table of quit episodes with months to relapse and relapse indicator
#   (2) the number of at-risk months (household-months where they purchased
#       last month)
# A "quit episode" is a month where the household purchased last month but not
# this month. Months to relapse is the number of months until the household
# next purchases (NA if censored).
compute_relapse_episodes <- function(dt_input, purchase_col)
{
  dt_work <- copy(dt_input)
  setkeyv(dt_work, c("household_code", "purchase_month"))

  # Lag the purchase indicator within each household
  dt_work[, lag_purchase := shift(get(purchase_col)), by = household_code]
  dt_work <- dt_work[!is.na(lag_purchase)]

  # Sequential observation index within each household
  dt_work[, obs_idx := seq_len(.N), by = household_code]

  # Number of at-risk months: household-months where they purchased last month
  n_at_risk <- sum(dt_work$lag_purchase == 1)

  # Quit episode: purchased last month but not this month
  dt_work[, quit_episode := fifelse(lag_purchase == 1 & get(purchase_col) == 0, 1, 0)]

  # For each observation, find the obs_idx of the next purchase within the
  # same household using reverse LOCF (next observation carried backward).
  # Rows with a purchase anchor their own obs_idx; rows without a purchase
  # get filled with the next obs_idx where a purchase occurs.
  # If no future purchase exists, the value remains NA (censored).
  dt_work[, next_purchase_idx := {
    x <- fifelse(get(purchase_col) == 1, obs_idx, NA_integer_)
    nafill(x, type = "nocb")
  }, by = household_code]

  # Months from quit to next purchase
  dt_work[, months_to_relapse := next_purchase_idx - obs_idx]

  # Extract quit episodes
  dt_episodes <- dt_work[quit_episode == 1, .(
    household_code,
    months_to_relapse,
    relapsed = fifelse(!is.na(months_to_relapse), 1, 0)
  )]

  # Return episodes and the at-risk denominator
  list(episodes = dt_episodes, n_at_risk = n_at_risk)
}


#############################
# Relapse analysis:
# quit-and-restart patterns
#############################

# Copy data and create product-specific purchase indicators
dt_relapse <- copy(dt)
dt_relapse[, any_cig  := fifelse(cig == 1 | cig_ecig == 1, 1, 0)]
dt_relapse[, any_ecig := fifelse(ecig == 1 | cig_ecig == 1, 1, 0)]

# Compute relapse episodes for each product type
list_cig  <- compute_relapse_episodes(dt_relapse, "any_cig")
list_ecig <- compute_relapse_episodes(dt_relapse, "any_ecig")

# Extract episode tables and at-risk counts
dt_episodes_cig  <- list_cig$episodes
dt_episodes_ecig <- list_ecig$episodes

# Tag each with product type and stack
dt_episodes_cig[,  product := "Cigarettes"]
dt_episodes_ecig[, product := "E-Cigarettes"]
dt_episodes <- rbindlist(list(dt_episodes_cig, dt_episodes_ecig))

# At-risk months (household-months where they purchased last month) by product
dt_at_risk <- data.table(
  product   = c("Cigarettes", "E-Cigarettes"),
  n_at_risk = c(list_cig$n_at_risk, list_ecig$n_at_risk)
)


#############################
# Relapse summary statistics
#############################

# Overall quit episode counts, monthly quit rate, and relapse rate by product
dt_summary <- dt_episodes[, .(
  total_episodes = .N,
  relapsed       = sum(relapsed),
  censored       = sum(relapsed == 0),
  relapse_rate   = round(mean(relapsed) * 100, 2)
), keyby = product]

# Merge in at-risk months and compute the monthly quit rate
# (fraction of at-risk months that result in a quit episode)
dt_summary <- dt_at_risk[dt_summary, on = "product"]
dt_summary[, monthly_quit_rate := round(total_episodes / n_at_risk * 100, 2)]
dt_summary

# Among episodes that relapsed, fraction that relapsed within 1 or 3 months
dt_horizon <- rbindlist(lapply(c(1, 3), function(h)
{
  dt_episodes[relapsed == 1, .(
    horizon  = h,
    n_within = sum(months_to_relapse <= h),
    n_total  = .N,
    rate_pct = round(sum(months_to_relapse <= h) / .N * 100, 2)
  ), keyby = product]
}))
dt_horizon


#############################
# CDF figure data for paper
#############################

# Compute the unconditional empirical CDF of months to relapse for cigarettes
# and e-cigarettes. The denominator is all quit episodes (including censored
# ones), so the curves asymptote at the relapse rate rather than 1.
# This data is read by pgfplots in the LaTeX paper.
dt_cdf_cig  <- dt_episodes[product == "Cigarettes"   & relapsed == 1]
dt_cdf_ecig <- dt_episodes[product == "E-Cigarettes" & relapsed == 1]
n_cig_total  <- nrow(dt_episodes[product == "Cigarettes"])
n_ecig_total <- nrow(dt_episodes[product == "E-Cigarettes"])

# Build CDF at each integer month from 0 to the max observed
max_months <- max(dt_episodes[relapsed == 1, months_to_relapse])
dt_cdf <- data.table(months_to_relapse = 0:max_months)
dt_cdf[, cdf_cig  := sapply(months_to_relapse, function(h) sum(dt_cdf_cig$months_to_relapse  <= h) / n_cig_total)]
dt_cdf[, cdf_ecig := sapply(months_to_relapse, function(h) sum(dt_cdf_ecig$months_to_relapse <= h) / n_ecig_total)]

# Save to the figures directory for pgfplots
fig_dir <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper",
                     "4th_Year_Paper/4th_Year_Paper_Figures")
fwrite(dt_cdf, file.path(fig_dir, "08_Relapse_CDF.txt"))
