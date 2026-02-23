################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script tests for anticipatory behavior around TYA (teen/young adult)
# status transitions as reduced-form evidence for present bias. The 4-state
# TYA classification from 05_TYA_State_Transitions.R captures proximity to
# transition:
#
#   State 1: No TYA, stable (no child within 2 years of teen age)
#   State 2: No TYA, approaching (oldest child is 11-12)
#   State 3: TYA present, stable (youngest TYA member <= 23)
#   State 4: TYA present, ending soon (youngest TYA member >= 24)
#
# Under beta = 1 (exponential discounting), forward-looking households in
# anticipatory states (2, 4) should adjust flavored e-cig purchases BEFORE
# the transition occurs, since they anticipate the future utility change
# from lambda_2 * 1[flavored] * 1[TYA].
#
# Under beta < 1 (present bias), adjustment concentrates at or after the
# actual transition. Anticipatory states should show weaker responses.
#
# This script estimates:
#   Within TYA=1 — ending soon (state 4) vs stable (state 3)
#   Binned TYA member age — tests whether the state 4 vs 3 difference
#     reflects a smooth age gradient or is concentrated near the transition
#
# Sample: restricted to households that ever purchased e-cigs (ecig = 1 or
# cig_ecig = 1), since never-buyers have no channel through which TYA
# status changes affect their behavior.
################################################################################


#############################
# Preliminaries
#############################

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# Set working directory
wd <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper",
                "4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
setwd(wd)

# Load packages
pacman::p_load(data.table, fixest)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 20)

# Panel columns
id_col   <- "household_code"
time_col <- "purchase_month"


#############################
# Load data
#############################

# Household identifiers (one per household-month observation)
dt_hh <- fread(paste0("./Household_Codes.csv"))

# TYA indicator (one per household-month observation, row-aligned with dt_hh)
dt_tya <- fread(paste0("./Teen_Young_Adult.csv"))

# 4-state TYA classification (one per household-month, row-aligned with dt_hh)
dt_tya_state <- fread(paste0("./TYA_States.csv"))

# Product choices (one per household-month observation, row-aligned with dt_hh)
dt_choices <- fread(paste0("./Product_Choices.csv"))

# Category choices (one per household-month observation, row-aligned with dt_hh)
dt_cat <- fread(paste0("./Category_Choices.csv"))

# Consumption spaces for intensive margin (mL per alternative)
dt_consumption <- fread(paste0("./Consumption_Spaces.csv"))

# Raw panel data with household member ages (needed for TYA age bins)
wd_raw <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper",
                    "4th_Year_Paper_Data/HMS/2021-Onward",
                    "Tobacco_Panelists_Purchases_2021-Onward")
dt_raw <- fread(file.path(wd_raw,
                          "tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv"),
                select = c("household_code", "purchase_month",
                           "male_head_age", "female_head_age",
                           paste0("Member_", 1:7, "_Age")))


#############################
# Construct outcome variables
#############################

# Flavored e-cig column names (7 flavored ecig bins + 2 flavored bundles)
flav_ecig_cols   <- grep("^flav_ecig_", names(dt_choices), value = TRUE)
flav_bundle_cols <- grep("^bundle_flav_", names(dt_choices), value = TRUE)
flav_cols <- c(flav_ecig_cols, flav_bundle_cols)

# Purchased flavored e-cig indicator: 1 if any flavored alt chosen
mat_flav <- as.matrix(dt_choices[, ..flav_cols])
purchased_flav <- as.integer(rowSums(mat_flav) >= 1)

# Purchased cigarettes indicator (placebo outcome)
cig_cols <- grep("^cig_", names(dt_choices), value = TRUE)
mat_cig  <- as.matrix(dt_choices[, ..cig_cols])
purchased_cig <- as.integer(rowSums(mat_cig) >= 1)

# E-cig or bundle purchase indicator (for sample restriction)
purchased_ecig_any <- as.integer(dt_cat[, ecig] == 1 | dt_cat[, cig_ecig] == 1)

# Flavored e-cig mL consumed per household-month (intensive margin)
# Build consumption vector aligned to product choice columns
alt_names    <- names(dt_choices)
flav_ecig_ml <- numeric(length(alt_names))
names(flav_ecig_ml) <- alt_names
for (i in seq_along(alt_names))
{
  alt <- alt_names[i]

  if (grepl("^flav_ecig_", alt))
  {
    # Pure flavored e-cig alternatives
    flav_ecig_ml[i] <- dt_consumption[alternative == alt, consumption]
  } else if (grepl("^bundle_flav_", alt))
  {
    # Flavored bundle: e-cig component only
    ecig_name <- paste0(alt, "_ecig")
    flav_ecig_ml[i] <- dt_consumption[alternative == ecig_name, consumption]
  }
}

# Compute mL consumed via matrix multiplication
mat_choices <- as.matrix(dt_choices)
flav_ml_it  <- as.numeric(mat_choices %*% flav_ecig_ml)


#############################
# Compute youngest TYA
# member age
#############################

# All age columns (heads + non-head members)
age_cols <- c("male_head_age", "female_head_age", paste0("Member_", 1:7, "_Age"))

# Compute ages as matrix; treat age <= 0 as missing
mat_ages <- as.matrix(dt_raw[, ..age_cols])
mat_ages[!is.na(mat_ages) & mat_ages <= 0] <- NA

# Youngest member in TYA range (13-25 inclusive)
mat_tya_ages <- mat_ages
mat_tya_ages[is.na(mat_tya_ages) | mat_tya_ages < 13 | mat_tya_ages > 25] <- NA
list_tya_ages    <- lapply(seq_len(ncol(mat_tya_ages)), function(k) mat_tya_ages[, k])
youngest_tya_age <- do.call(pmin, c(list_tya_ages, na.rm = TRUE))
youngest_tya_age[is.infinite(youngest_tya_age)] <- NA_real_


#############################
# Build panel dataset
#############################

# Combine all variables into a single data.table
dt <- data.table(
  household_code   = dt_hh[, household_code],
  purchase_month   = dt_tya[, purchase_month],
  tya              = dt_tya[, teen_or_young_adult_present],
  tya_state        = dt_tya_state[, tya_state],
  purchased_flav   = purchased_flav,
  purchased_cig    = purchased_cig,
  purchased_ecig   = purchased_ecig_any,
  flav_ml          = flav_ml_it,
  youngest_tya_age = youngest_tya_age
)

# Restrict to households that ever purchased e-cigs (ecig = 1 or cig_ecig = 1)
dt[, ever_ecig := max(purchased_ecig), by = household_code]
dt <- dt[ever_ecig == 1]

# Create TYA state indicators
dt[, tya_approaching := as.integer(tya_state == 2)]
dt[, tya_ending_soon := as.integer(tya_state == 4)]


#############################
# Within
# TYA = 1 (ending soon vs
# stable)
#############################

# Restrict to TYA = 1 (states 3 and 4 only)
# Tests whether ending-soon households (state 4) differ from stable (state 3)
dt_tya1 <- dt[tya == 1]

# Extensive margin: flavored e-cig purchase indicator
est_flav_within1 <- feols(
  purchased_flav ~ tya_ending_soon | household_code + purchase_month,
  data    = dt_tya1,
  cluster = ~household_code
)
summary(est_flav_within1)

# Intensive margin: flavored e-cig mL consumed
est_ml_within1 <- feols(
  flav_ml ~ tya_ending_soon | household_code + purchase_month,
  data    = dt_tya1,
  cluster = ~household_code
)
summary(est_ml_within1)

# Placebo: cigarette purchases
est_cig_within1 <- feols(
  purchased_cig ~ tya_ending_soon | household_code + purchase_month,
  data    = dt_tya1,
  cluster = ~household_code
)
summary(est_cig_within1)


#############################
# Binned TYA member age
# (within TYA = 1)
#############################

# Tests whether the state 4 vs state 3 difference reflects a smooth age gradient
# (age composition effect) or is concentrated near the transition (anticipation).
# Under a pure age effect, flavored e-cig rates should decline monotonically as
# the youngest TYA member ages from 13 to 25. Under anticipation, rates should be
# flat from 13-22 and only drop at 23-25.
dt_tya1_age <- dt[tya == 1 & !is.na(youngest_tya_age)]

# Bin youngest TYA member age (reference = 13-16)
dt_tya1_age[, tya_age_bin := fcase(
  youngest_tya_age <= 16,                                    "13-16",
  youngest_tya_age >= 17 & youngest_tya_age <= 19,           "17-19",
  youngest_tya_age >= 20 & youngest_tya_age <= 22,           "20-22",
  youngest_tya_age >= 23 & youngest_tya_age <= 25,           "23-25"
)]
dt_tya1_age[, tya_age_bin := factor(tya_age_bin,
                                     levels = c("13-16", "17-19", "20-22", "23-25"))]

# Bin distribution
dt_tya1_age[, .N, keyby = tya_age_bin]

# Extensive margin: flavored e-cig purchase by age bin (reference = 13-16)
est_flav_binned <- feols(
  purchased_flav ~ tya_age_bin | household_code + purchase_month,
  data    = dt_tya1_age,
  cluster = ~household_code
)
summary(est_flav_binned)

# Placebo: cigarette purchases by age bin
est_cig_binned <- feols(
  purchased_cig ~ tya_age_bin | household_code + purchase_month,
  data    = dt_tya1_age,
  cluster = ~household_code
)
summary(est_cig_binned)





















