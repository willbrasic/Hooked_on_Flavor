################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script prepares the 2023 validation data for out-of-sample model
# evaluation. It is the companion to 01_Data_Prep_Holdout.R, which prepares
# the 2021-2022 estimation data.
#
# Strategy: temporal hold-out.
#   - Estimation sample: ALL households x 2021-2022      
#   - Validation sample: overlap households x 2023       
#
# Overlap households are those present in both 2021-2022 AND 2023.
#
# The 46-alternative structure (bins, consumption medians, nicotine medians)
# must be identical to what was used during estimation. Bundle split thresholds
# are therefore computed from the 2021-2022 data and applied to 2023 choices.
# Consumption_Spaces, Nicotine_Spaces, Pricing_Spaces, Halton_Draw_Transitions,
# and Halton_Draw_Shocks are copied from Data_Holdout/ unchanged.
################################################################################


#############################
# Preliminaries
#############################

rm(list = ls())
graphics.off()
cat("\014")

wd <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/",
                "4th_Year_Paper_Data/HMS/2021-Onward/Tobacco_Panelists_Purchases_2021-Onward/")
setwd(wd)

pacman::p_load(data.table)

options(scipen = 999)
options(max.print = 999999)
options(datatable.print.nrows = 200)

# Load full panel
file_name <- "./all_panelists_purchases_monthly_CLEANED_2021-Onward.csv"
dt_full <- fread(file_name)

# Output directories
dir_holdout     <- "../Dynamic_Model/Data_Holdout/"
dir_holdout_val <- "../Dynamic_Model/Data_Holdout_Val/"
if (!dir.exists(dir_holdout_val))
{
  dir.create(dir_holdout_val)
}


#############################
# Identify Overlap Households
#############################

dt_full[, year_pm := year(purchase_month)]

cat("Months in data:", format(sort(unique(dt_full[, purchase_month])), "%Y-%m"), "\n")

list_hh_est    <- dt_full[year_pm %in% c(2021, 2022), unique(household_code)]
list_hh_val    <- dt_full[year_pm == 2023,              unique(household_code)]
list_overlap_hh <- intersect(list_hh_est, list_hh_val)

cat("Total households in 2021-2022:         ", length(list_hh_est),     "\n")
cat("Total households in 2023:              ", length(list_hh_val),     "\n")
cat("Overlap households (in both windows):  ", length(list_overlap_hh), "\n")

# 2021-2022 data: used only to compute bundle split thresholds (full estimation sample, not overlap only)
dt_est <- dt_full[year_pm %in% c(2021, 2022)]

# 2023 data: the validation sample
dt <- dt_full[household_code %in% list_overlap_hh & year_pm == 2023]
dt[, year_pm := NULL]

cat("Validation observations (overlap HHs x 2023):", nrow(dt), "\n")

rm(dt_full, list_hh_est, list_hh_val)


#############################
# Copy Static Files from
# Data_Holdout/
#############################

static_files <- c(
  "Overlap_Household_Codes.csv",
  "Consumption_Spaces.csv",
  "Nicotine_Spaces.csv",
  "Pricing_Spaces.csv",
  "Halton_Draw_Transitions.csv",
  "Halton_Draw_Shocks.csv",
  "Median_Per-Unit_Monthly_Prices.csv"
)

for (f in static_files)
{
  src  <- paste0(dir_holdout, f)
  dest <- paste0(dir_holdout_val, f)
  if (file.exists(src))
  {
    file.copy(src, dest, overwrite = TRUE)
    cat("Copied:", f, "\n")
  } else
  {
    cat("Warning: source file not found:", src, "\n")
  }
}


#############################
# Bundle Split Thresholds
# (from 2021-2022 estimation
#  data for consistency)
#############################

med_cig_packs <- dt_est[total_packs > 0 & total_mL == 0,          median(total_packs,               na.rm = TRUE)]
med_orig_mL   <- dt_est[total_original_mL > 0 & total_packs == 0, median(total_original_mL,          na.rm = TRUE)]
med_nfda_mL   <- dt_est[total_non_fda_flavored_mL > 0 & total_packs == 0, median(total_non_fda_flavored_mL, na.rm = TRUE)]
med_fda_mL    <- dt_est[total_fda_flavored_mL > 0 & total_packs == 0,     median(total_fda_flavored_mL,     na.rm = TRUE)]

cat("Bundle split thresholds (from 2021-2022):\n")
cat("  Cig packs median:          ", med_cig_packs, "\n")
cat("  Original ecig mL median:   ", med_orig_mL,   "\n")
cat("  Non-FDA flavored mL median:", med_nfda_mL,   "\n")
cat("  FDA flavored mL median:    ", med_fda_mL,    "\n")

rm(dt_est)


#############################
# Household Codes
#############################

dt_hh <- data.table("household_code" = dt[, household_code])

file_name <- paste0(dir_holdout_val, "Household_Codes.csv")
fwrite(dt_hh, file_name)

if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Category Level Choices
#############################

dt_category_choices <- dt[, .(
  outside_option,
  cig,
  ecig,
  cig_ecig
)]

stopifnot("NA values found in category choice indicators" = !any(is.na(dt_category_choices)))
all(rowSums(dt_category_choices) == 1)

file_name <- paste0(dir_holdout_val, "Category_Choices.csv")
fwrite(dt_category_choices, file_name)

if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Product Level Choices
#############################

dt_product_choices <- dt[, .(

  # Outside option
  outside_option = fifelse(cig == 0 & ecig == 0 & cig_ecig == 0, 1, 0),

  # Cigarette quantity alternatives (cigs only: total_mL == 0)
  cig_1        = fifelse(total_packs == 1  & total_mL == 0, 1, 0),
  cig_2        = fifelse(total_packs == 2  & total_mL == 0, 1, 0),
  cig_3to4     = fifelse(total_packs >= 3  & total_packs <= 4  & total_mL == 0, 1, 0),
  cig_5to9     = fifelse(total_packs >= 5  & total_packs <= 9  & total_mL == 0, 1, 0),
  cig_10       = fifelse(total_packs == 10 & total_mL == 0, 1, 0),
  cig_11to19   = fifelse(total_packs >= 11 & total_packs <= 19 & total_mL == 0, 1, 0),
  cig_20       = fifelse(total_packs == 20 & total_mL == 0, 1, 0),
  cig_21to29   = fifelse(total_packs >= 21 & total_packs <= 29 & total_mL == 0, 1, 0),
  cig_30       = fifelse(total_packs == 30 & total_mL == 0, 1, 0),
  cig_31to39   = fifelse(total_packs >= 31 & total_packs <= 39 & total_mL == 0, 1, 0),
  cig_40       = fifelse(total_packs == 40 & total_mL == 0, 1, 0),
  cig_41plus   = fifelse(total_packs >= 41 & total_mL == 0, 1, 0),

  # Original e-cigarette alternatives (e-cig only: total_packs == 0)
  orig_ecig_0to5    = fifelse(total_original_mL > 0  & total_original_mL <= 5  & total_packs == 0, 1, 0),
  orig_ecig_5to10   = fifelse(total_original_mL > 5  & total_original_mL <= 10 & total_packs == 0, 1, 0),
  orig_ecig_10to15  = fifelse(total_original_mL > 10 & total_original_mL <= 15 & total_packs == 0, 1, 0),
  orig_ecig_15to20  = fifelse(total_original_mL > 15 & total_original_mL <= 20 & total_packs == 0, 1, 0),
  orig_ecig_20to30  = fifelse(total_original_mL > 20 & total_original_mL <= 30 & total_packs == 0, 1, 0),
  orig_ecig_30to50  = fifelse(total_original_mL > 30 & total_original_mL <= 50 & total_packs == 0, 1, 0),
  orig_ecig_50plus  = fifelse(total_original_mL > 50 & total_packs == 0, 1, 0),

  # Non-FDA flavored e-cigarette alternatives (e-cig only: total_packs == 0)
  non_fda_flav_ecig_0to5    = fifelse(total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= 5  & total_packs == 0, 1, 0),
  non_fda_flav_ecig_5to10   = fifelse(total_non_fda_flavored_mL > 5  & total_non_fda_flavored_mL <= 10 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_10to15  = fifelse(total_non_fda_flavored_mL > 10 & total_non_fda_flavored_mL <= 15 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_15to20  = fifelse(total_non_fda_flavored_mL > 15 & total_non_fda_flavored_mL <= 20 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_20to30  = fifelse(total_non_fda_flavored_mL > 20 & total_non_fda_flavored_mL <= 30 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_30to50  = fifelse(total_non_fda_flavored_mL > 30 & total_non_fda_flavored_mL <= 50 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_50plus  = fifelse(total_non_fda_flavored_mL > 50 & total_packs == 0, 1, 0),

  # FDA-authorized flavored e-cigarette alternatives (e-cig only: total_packs == 0)
  fda_flav_ecig_0to5    = fifelse(total_fda_flavored_mL > 0  & total_fda_flavored_mL <= 5  & total_packs == 0, 1, 0),
  fda_flav_ecig_5to10   = fifelse(total_fda_flavored_mL > 5  & total_fda_flavored_mL <= 10 & total_packs == 0, 1, 0),
  fda_flav_ecig_10to15  = fifelse(total_fda_flavored_mL > 10 & total_fda_flavored_mL <= 15 & total_packs == 0, 1, 0),
  fda_flav_ecig_15to20  = fifelse(total_fda_flavored_mL > 15 & total_fda_flavored_mL <= 20 & total_packs == 0, 1, 0),
  fda_flav_ecig_20to30  = fifelse(total_fda_flavored_mL > 20 & total_fda_flavored_mL <= 30 & total_packs == 0, 1, 0),
  fda_flav_ecig_30to50  = fifelse(total_fda_flavored_mL > 30 & total_fda_flavored_mL <= 50 & total_packs == 0, 1, 0),
  fda_flav_ecig_50plus  = fifelse(total_fda_flavored_mL > 50 & total_packs == 0, 1, 0),

  # Bundles: 4 per flavor type (lo/hi cig x lo/hi ecig at 2021-2022 thresholds)
  bundle_orig_ll = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL, 1, 0),
  bundle_orig_lh = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL, 1, 0),
  bundle_orig_hl = fifelse(total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL, 1, 0),
  bundle_orig_hh = fifelse(total_packs > med_cig_packs & total_original_mL > med_orig_mL, 1, 0),
  bundle_nfda_ll = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL, 1, 0),
  bundle_nfda_lh = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL, 1, 0),
  bundle_nfda_hl = fifelse(total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL, 1, 0),
  bundle_nfda_hh = fifelse(total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL, 1, 0),
  bundle_fda_ll  = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL, 1, 0),
  bundle_fda_lh  = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL, 1, 0),
  bundle_fda_hl  = fifelse(total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL, 1, 0),
  bundle_fda_hh  = fifelse(total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL, 1, 0)
)]

data.table(
  alternative = names(dt_product_choices),
  fraction = colMeans(dt_product_choices)
)[order(-fraction)]

stopifnot("NA values found in product choice indicators" = !any(is.na(dt_product_choices)))
all(rowSums(dt_product_choices) == 1)

file_name <- paste0(dir_holdout_val, "Product_Choices.csv")
fwrite(dt_product_choices, file_name)

if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Youth Indicator
#############################

dt_tya <- dt[, .(household_code, purchase_month, teen_or_young_adult_present)]
setnames(dt_tya, "teen_or_young_adult_present", "tya_state")

file_name <- paste0(dir_holdout_val, "TYA_States.csv")
fwrite(dt_tya, file_name)

if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Prices
#############################

# County-month median per-unit prices BY BIN
dt_county_month_prices <- dt[, .(

  cig_1_cm      = median(real_per_pack_price_paid[total_packs == 1 ], na.rm = TRUE),
  cig_2_cm      = median(real_per_pack_price_paid[total_packs == 2 ], na.rm = TRUE),
  cig_3to4_cm   = median(real_per_pack_price_paid[total_packs >= 3  & total_packs <= 4 ], na.rm = TRUE),
  cig_5to9_cm   = median(real_per_pack_price_paid[total_packs >= 5  & total_packs <= 9 ], na.rm = TRUE),
  cig_10_cm     = median(real_per_pack_price_paid[total_packs == 10], na.rm = TRUE),
  cig_11to19_cm = median(real_per_pack_price_paid[total_packs >= 11 & total_packs <= 19], na.rm = TRUE),
  cig_20_cm     = median(real_per_pack_price_paid[total_packs == 20], na.rm = TRUE),
  cig_21to29_cm = median(real_per_pack_price_paid[total_packs >= 21 & total_packs <= 29], na.rm = TRUE),
  cig_30_cm     = median(real_per_pack_price_paid[total_packs == 30], na.rm = TRUE),
  cig_31to39_cm = median(real_per_pack_price_paid[total_packs >= 31 & total_packs <= 39], na.rm = TRUE),
  cig_40_cm     = median(real_per_pack_price_paid[total_packs == 40], na.rm = TRUE),
  cig_41plus_cm = median(real_per_pack_price_paid[total_packs >= 41], na.rm = TRUE),

  ecig_0to5_cm   = median(real_per_mL_price_paid[total_mL > 0  & total_mL <= 5],  na.rm = TRUE),
  ecig_5to10_cm  = median(real_per_mL_price_paid[total_mL > 5  & total_mL <= 10], na.rm = TRUE),
  ecig_10to15_cm = median(real_per_mL_price_paid[total_mL > 10 & total_mL <= 15], na.rm = TRUE),
  ecig_15to20_cm = median(real_per_mL_price_paid[total_mL > 15 & total_mL <= 20], na.rm = TRUE),
  ecig_20to30_cm = median(real_per_mL_price_paid[total_mL > 20 & total_mL <= 30], na.rm = TRUE),
  ecig_30to50_cm = median(real_per_mL_price_paid[total_mL > 30 & total_mL <= 50], na.rm = TRUE),
  ecig_50plus_cm = median(real_per_mL_price_paid[total_mL > 50], na.rm = TRUE),

  bundle_lo_cig_cm       = median(real_per_pack_price_paid[total_packs > 0  & total_packs <= med_cig_packs], na.rm = TRUE),
  bundle_hi_cig_cm       = median(real_per_pack_price_paid[total_packs > med_cig_packs],                     na.rm = TRUE),
  bundle_orig_lo_ecig_cm = median(real_per_mL_price_paid[total_original_mL > 0       & total_original_mL <= med_orig_mL & total_packs == 0], na.rm = TRUE),
  bundle_orig_hi_ecig_cm = median(real_per_mL_price_paid[total_original_mL > med_orig_mL & total_packs == 0],                                na.rm = TRUE),
  bundle_nfda_lo_ecig_cm = median(real_per_mL_price_paid[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL & total_packs == 0], na.rm = TRUE),
  bundle_nfda_hi_ecig_cm = median(real_per_mL_price_paid[total_non_fda_flavored_mL > med_nfda_mL & total_packs == 0],                                    na.rm = TRUE),
  bundle_fda_lo_ecig_cm  = median(real_per_mL_price_paid[total_fda_flavored_mL > 0   & total_fda_flavored_mL <= med_fda_mL & total_packs == 0],          na.rm = TRUE),
  bundle_fda_hi_ecig_cm  = median(real_per_mL_price_paid[total_fda_flavored_mL > med_fda_mL & total_packs == 0],                                         na.rm = TRUE)

), keyby = .(fips_county_code, purchase_month)]

# State-month median per-unit prices BY BIN
dt_state_month_prices <- dt[, .(

  cig_1_sm      = median(real_per_pack_price_paid[total_packs == 1 ], na.rm = TRUE),
  cig_2_sm      = median(real_per_pack_price_paid[total_packs == 2 ], na.rm = TRUE),
  cig_3to4_sm   = median(real_per_pack_price_paid[total_packs >= 3  & total_packs <= 4 ], na.rm = TRUE),
  cig_5to9_sm   = median(real_per_pack_price_paid[total_packs >= 5  & total_packs <= 9 ], na.rm = TRUE),
  cig_10_sm     = median(real_per_pack_price_paid[total_packs == 10], na.rm = TRUE),
  cig_11to19_sm = median(real_per_pack_price_paid[total_packs >= 11 & total_packs <= 19], na.rm = TRUE),
  cig_20_sm     = median(real_per_pack_price_paid[total_packs == 20], na.rm = TRUE),
  cig_21to29_sm = median(real_per_pack_price_paid[total_packs >= 21 & total_packs <= 29], na.rm = TRUE),
  cig_30_sm     = median(real_per_pack_price_paid[total_packs == 30], na.rm = TRUE),
  cig_31to39_sm = median(real_per_pack_price_paid[total_packs >= 31 & total_packs <= 39], na.rm = TRUE),
  cig_40_sm     = median(real_per_pack_price_paid[total_packs == 40], na.rm = TRUE),
  cig_41plus_sm = median(real_per_pack_price_paid[total_packs >= 41], na.rm = TRUE),

  ecig_0to5_sm   = median(real_per_mL_price_paid[total_mL > 0  & total_mL <= 5],  na.rm = TRUE),
  ecig_5to10_sm  = median(real_per_mL_price_paid[total_mL > 5  & total_mL <= 10], na.rm = TRUE),
  ecig_10to15_sm = median(real_per_mL_price_paid[total_mL > 10 & total_mL <= 15], na.rm = TRUE),
  ecig_15to20_sm = median(real_per_mL_price_paid[total_mL > 15 & total_mL <= 20], na.rm = TRUE),
  ecig_20to30_sm = median(real_per_mL_price_paid[total_mL > 20 & total_mL <= 30], na.rm = TRUE),
  ecig_30to50_sm = median(real_per_mL_price_paid[total_mL > 30 & total_mL <= 50], na.rm = TRUE),
  ecig_50plus_sm = median(real_per_mL_price_paid[total_mL > 50], na.rm = TRUE),

  bundle_lo_cig_sm       = median(real_per_pack_price_paid[total_packs > 0  & total_packs <= med_cig_packs], na.rm = TRUE),
  bundle_hi_cig_sm       = median(real_per_pack_price_paid[total_packs > med_cig_packs],                     na.rm = TRUE),
  bundle_orig_lo_ecig_sm = median(real_per_mL_price_paid[total_original_mL > 0       & total_original_mL <= med_orig_mL & total_packs == 0], na.rm = TRUE),
  bundle_orig_hi_ecig_sm = median(real_per_mL_price_paid[total_original_mL > med_orig_mL & total_packs == 0],                                na.rm = TRUE),
  bundle_nfda_lo_ecig_sm = median(real_per_mL_price_paid[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL & total_packs == 0], na.rm = TRUE),
  bundle_nfda_hi_ecig_sm = median(real_per_mL_price_paid[total_non_fda_flavored_mL > med_nfda_mL & total_packs == 0],                                    na.rm = TRUE),
  bundle_fda_lo_ecig_sm  = median(real_per_mL_price_paid[total_fda_flavored_mL > 0   & total_fda_flavored_mL <= med_fda_mL & total_packs == 0],          na.rm = TRUE),
  bundle_fda_hi_ecig_sm  = median(real_per_mL_price_paid[total_fda_flavored_mL > med_fda_mL & total_packs == 0],                                         na.rm = TRUE)

), keyby = .(fips_state_code, purchase_month)]

# Monthly median per-unit prices BY BIN (fallback)
dt_month_prices <- dt[, .(

  cig_1_m      = median(real_per_pack_price_paid[total_packs == 1 ], na.rm = TRUE),
  cig_2_m      = median(real_per_pack_price_paid[total_packs == 2 ], na.rm = TRUE),
  cig_3to4_m   = median(real_per_pack_price_paid[total_packs >= 3  & total_packs <= 4 ], na.rm = TRUE),
  cig_5to9_m   = median(real_per_pack_price_paid[total_packs >= 5  & total_packs <= 9 ], na.rm = TRUE),
  cig_10_m     = median(real_per_pack_price_paid[total_packs == 10], na.rm = TRUE),
  cig_11to19_m = median(real_per_pack_price_paid[total_packs >= 11 & total_packs <= 19], na.rm = TRUE),
  cig_20_m     = median(real_per_pack_price_paid[total_packs == 20], na.rm = TRUE),
  cig_21to29_m = median(real_per_pack_price_paid[total_packs >= 21 & total_packs <= 29], na.rm = TRUE),
  cig_30_m     = median(real_per_pack_price_paid[total_packs == 30], na.rm = TRUE),
  cig_31to39_m = median(real_per_pack_price_paid[total_packs >= 31 & total_packs <= 39], na.rm = TRUE),
  cig_40_m     = median(real_per_pack_price_paid[total_packs == 40], na.rm = TRUE),
  cig_41plus_m = median(real_per_pack_price_paid[total_packs >= 41], na.rm = TRUE),

  ecig_0to5_m   = median(real_per_mL_price_paid[total_mL > 0  & total_mL <= 5],  na.rm = TRUE),
  ecig_5to10_m  = median(real_per_mL_price_paid[total_mL > 5  & total_mL <= 10], na.rm = TRUE),
  ecig_10to15_m = median(real_per_mL_price_paid[total_mL > 10 & total_mL <= 15], na.rm = TRUE),
  ecig_15to20_m = median(real_per_mL_price_paid[total_mL > 15 & total_mL <= 20], na.rm = TRUE),
  ecig_20to30_m = median(real_per_mL_price_paid[total_mL > 20 & total_mL <= 30], na.rm = TRUE),
  ecig_30to50_m = median(real_per_mL_price_paid[total_mL > 30 & total_mL <= 50], na.rm = TRUE),
  ecig_50plus_m = median(real_per_mL_price_paid[total_mL > 50], na.rm = TRUE),

  bundle_lo_cig_m       = median(real_per_pack_price_paid[total_packs > 0  & total_packs <= med_cig_packs], na.rm = TRUE),
  bundle_hi_cig_m       = median(real_per_pack_price_paid[total_packs > med_cig_packs],                     na.rm = TRUE),
  bundle_orig_lo_ecig_m = median(real_per_mL_price_paid[total_original_mL > 0       & total_original_mL <= med_orig_mL & total_packs == 0], na.rm = TRUE),
  bundle_orig_hi_ecig_m = median(real_per_mL_price_paid[total_original_mL > med_orig_mL & total_packs == 0],                                na.rm = TRUE),
  bundle_nfda_lo_ecig_m = median(real_per_mL_price_paid[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL & total_packs == 0], na.rm = TRUE),
  bundle_nfda_hi_ecig_m = median(real_per_mL_price_paid[total_non_fda_flavored_mL > med_nfda_mL & total_packs == 0],                                    na.rm = TRUE),
  bundle_fda_lo_ecig_m  = median(real_per_mL_price_paid[total_fda_flavored_mL > 0   & total_fda_flavored_mL <= med_fda_mL & total_packs == 0],          na.rm = TRUE),
  bundle_fda_hi_ecig_m  = median(real_per_mL_price_paid[total_fda_flavored_mL > med_fda_mL & total_packs == 0],                                         na.rm = TRUE)

), keyby = purchase_month]

nrow(dt_month_prices[!complete.cases(dt_month_prices)])

dt_prices <- dt[, .(household_code, purchase_month, fips_state_code, fips_county_code,
                    total_packs, total_mL, total_flavored_mL, total_original_mL,
                    real_per_pack_price_paid, real_per_mL_price_paid)]
dt_prices[, names(dt_product_choices) := dt_product_choices]

# Merge county-month median prices
cm_cols <- setdiff(names(dt_county_month_prices), c("fips_county_code", "purchase_month"))
dt_prices[
  dt_county_month_prices,
  on = .(fips_county_code, purchase_month),
  (cm_cols) := mget(paste0("i.", cm_cols))
]

# Merge state-month median prices
sm_cols <- setdiff(names(dt_state_month_prices), c("fips_state_code", "purchase_month"))
dt_prices[
  dt_state_month_prices,
  on = .(fips_state_code, purchase_month),
  (sm_cols) := mget(paste0("i.", sm_cols))
]

# Merge monthly median prices
m_cols <- setdiff(names(dt_month_prices), "purchase_month")
dt_prices[
  dt_month_prices,
  on = .(purchase_month),
  (m_cols) := mget(paste0("i.", m_cols))
]

# Tier 1: actual prices paid
dt_prices[, cig_1_actual      := fifelse(cig_1 == 1,      real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_2_actual      := fifelse(cig_2 == 1,      real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_3to4_actual   := fifelse(cig_3to4 == 1,   real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_5to9_actual   := fifelse(cig_5to9 == 1,   real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_10_actual     := fifelse(cig_10 == 1,     real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_11to19_actual := fifelse(cig_11to19 == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_20_actual     := fifelse(cig_20 == 1,     real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_21to29_actual := fifelse(cig_21to29 == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_30_actual     := fifelse(cig_30 == 1,     real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_31to39_actual := fifelse(cig_31to39 == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_40_actual     := fifelse(cig_40 == 1,     real_per_pack_price_paid, NA_real_)]
dt_prices[, cig_41plus_actual := fifelse(cig_41plus == 1, real_per_pack_price_paid, NA_real_)]

dt_prices[, ecig_0to5_actual   := fifelse(total_mL > 0  & total_mL <= 5,  real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_5to10_actual  := fifelse(total_mL > 5  & total_mL <= 10, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_10to15_actual := fifelse(total_mL > 10 & total_mL <= 15, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_15to20_actual := fifelse(total_mL > 15 & total_mL <= 20, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_20to30_actual := fifelse(total_mL > 20 & total_mL <= 30, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_30to50_actual := fifelse(total_mL > 30 & total_mL <= 50, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_50plus_actual := fifelse(total_mL > 50,                  real_per_mL_price_paid, NA_real_)]

dt_prices[, bundle_orig_ll_cig_actual  := fifelse(bundle_orig_ll == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_orig_ll_ecig_actual := fifelse(bundle_orig_ll == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_orig_lh_cig_actual  := fifelse(bundle_orig_lh == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_orig_lh_ecig_actual := fifelse(bundle_orig_lh == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_orig_hl_cig_actual  := fifelse(bundle_orig_hl == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_orig_hl_ecig_actual := fifelse(bundle_orig_hl == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_orig_hh_cig_actual  := fifelse(bundle_orig_hh == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_orig_hh_ecig_actual := fifelse(bundle_orig_hh == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_nfda_ll_cig_actual  := fifelse(bundle_nfda_ll == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_nfda_ll_ecig_actual := fifelse(bundle_nfda_ll == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_nfda_lh_cig_actual  := fifelse(bundle_nfda_lh == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_nfda_lh_ecig_actual := fifelse(bundle_nfda_lh == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_nfda_hl_cig_actual  := fifelse(bundle_nfda_hl == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_nfda_hl_ecig_actual := fifelse(bundle_nfda_hl == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_nfda_hh_cig_actual  := fifelse(bundle_nfda_hh == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_nfda_hh_ecig_actual := fifelse(bundle_nfda_hh == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_fda_ll_cig_actual   := fifelse(bundle_fda_ll  == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_fda_ll_ecig_actual  := fifelse(bundle_fda_ll  == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_fda_lh_cig_actual   := fifelse(bundle_fda_lh  == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_fda_lh_ecig_actual  := fifelse(bundle_fda_lh  == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_fda_hl_cig_actual   := fifelse(bundle_fda_hl  == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_fda_hl_ecig_actual  := fifelse(bundle_fda_hl  == 1, real_per_mL_price_paid,   NA_real_)]
dt_prices[, bundle_fda_hh_cig_actual   := fifelse(bundle_fda_hh  == 1, real_per_pack_price_paid, NA_real_)]
dt_prices[, bundle_fda_hh_ecig_actual  := fifelse(bundle_fda_hh  == 1, real_per_mL_price_paid,   NA_real_)]

# Final prices: fcoalesce(actual, county-month, state-month, monthly)
dt_prices[, cig_1_p      := fcoalesce(cig_1_actual, cig_1_cm, cig_1_sm, cig_1_m)]
dt_prices[, cig_2_p      := fcoalesce(cig_2_actual, cig_2_cm, cig_2_sm, cig_2_m)]
dt_prices[, cig_3to4_p   := fcoalesce(cig_3to4_actual, cig_3to4_cm, cig_3to4_sm, cig_3to4_m)]
dt_prices[, cig_5to9_p   := fcoalesce(cig_5to9_actual, cig_5to9_cm, cig_5to9_sm, cig_5to9_m)]
dt_prices[, cig_10_p     := fcoalesce(cig_10_actual, cig_10_cm, cig_10_sm, cig_10_m)]
dt_prices[, cig_11to19_p := fcoalesce(cig_11to19_actual, cig_11to19_cm, cig_11to19_sm, cig_11to19_m)]
dt_prices[, cig_20_p     := fcoalesce(cig_20_actual, cig_20_cm, cig_20_sm, cig_20_m)]
dt_prices[, cig_21to29_p := fcoalesce(cig_21to29_actual, cig_21to29_cm, cig_21to29_sm, cig_21to29_m)]
dt_prices[, cig_30_p     := fcoalesce(cig_30_actual, cig_30_cm, cig_30_sm, cig_30_m)]
dt_prices[, cig_31to39_p := fcoalesce(cig_31to39_actual, cig_31to39_cm, cig_31to39_sm, cig_31to39_m)]
dt_prices[, cig_40_p     := fcoalesce(cig_40_actual, cig_40_cm, cig_40_sm, cig_40_m)]
dt_prices[, cig_41plus_p := fcoalesce(cig_41plus_actual, cig_41plus_cm, cig_41plus_sm, cig_41plus_m)]

dt_prices[, ecig_0to5_p   := fcoalesce(ecig_0to5_actual, ecig_0to5_cm, ecig_0to5_sm, ecig_0to5_m)]
dt_prices[, ecig_5to10_p  := fcoalesce(ecig_5to10_actual, ecig_5to10_cm, ecig_5to10_sm, ecig_5to10_m)]
dt_prices[, ecig_10to15_p := fcoalesce(ecig_10to15_actual, ecig_10to15_cm, ecig_10to15_sm, ecig_10to15_m)]
dt_prices[, ecig_15to20_p := fcoalesce(ecig_15to20_actual, ecig_15to20_cm, ecig_15to20_sm, ecig_15to20_m)]
dt_prices[, ecig_20to30_p := fcoalesce(ecig_20to30_actual, ecig_20to30_cm, ecig_20to30_sm, ecig_20to30_m)]
dt_prices[, ecig_30to50_p := fcoalesce(ecig_30to50_actual, ecig_30to50_cm, ecig_30to50_sm, ecig_30to50_m)]
dt_prices[, ecig_50plus_p := fcoalesce(ecig_50plus_actual, ecig_50plus_cm, ecig_50plus_sm, ecig_50plus_m)]

dt_prices[, bundle_orig_ll_cig_p  := fcoalesce(bundle_orig_ll_cig_actual,  bundle_lo_cig_cm, bundle_lo_cig_sm, bundle_lo_cig_m)]
dt_prices[, bundle_orig_ll_ecig_p := fcoalesce(bundle_orig_ll_ecig_actual, bundle_orig_lo_ecig_cm, bundle_orig_lo_ecig_sm, bundle_orig_lo_ecig_m)]
dt_prices[, bundle_orig_lh_cig_p  := fcoalesce(bundle_orig_lh_cig_actual,  bundle_lo_cig_cm, bundle_lo_cig_sm, bundle_lo_cig_m)]
dt_prices[, bundle_orig_lh_ecig_p := fcoalesce(bundle_orig_lh_ecig_actual, bundle_orig_hi_ecig_cm, bundle_orig_hi_ecig_sm, bundle_orig_hi_ecig_m)]
dt_prices[, bundle_orig_hl_cig_p  := fcoalesce(bundle_orig_hl_cig_actual,  bundle_hi_cig_cm, bundle_hi_cig_sm, bundle_hi_cig_m)]
dt_prices[, bundle_orig_hl_ecig_p := fcoalesce(bundle_orig_hl_ecig_actual, bundle_orig_lo_ecig_cm, bundle_orig_lo_ecig_sm, bundle_orig_lo_ecig_m)]
dt_prices[, bundle_orig_hh_cig_p  := fcoalesce(bundle_orig_hh_cig_actual,  bundle_hi_cig_cm, bundle_hi_cig_sm, bundle_hi_cig_m)]
dt_prices[, bundle_orig_hh_ecig_p := fcoalesce(bundle_orig_hh_ecig_actual, bundle_orig_hi_ecig_cm, bundle_orig_hi_ecig_sm, bundle_orig_hi_ecig_m)]
dt_prices[, bundle_nfda_ll_cig_p  := fcoalesce(bundle_nfda_ll_cig_actual,  bundle_lo_cig_cm, bundle_lo_cig_sm, bundle_lo_cig_m)]
dt_prices[, bundle_nfda_ll_ecig_p := fcoalesce(bundle_nfda_ll_ecig_actual, bundle_nfda_lo_ecig_cm, bundle_nfda_lo_ecig_sm, bundle_nfda_lo_ecig_m)]
dt_prices[, bundle_nfda_lh_cig_p  := fcoalesce(bundle_nfda_lh_cig_actual,  bundle_lo_cig_cm, bundle_lo_cig_sm, bundle_lo_cig_m)]
dt_prices[, bundle_nfda_lh_ecig_p := fcoalesce(bundle_nfda_lh_ecig_actual, bundle_nfda_hi_ecig_cm, bundle_nfda_hi_ecig_sm, bundle_nfda_hi_ecig_m)]
dt_prices[, bundle_nfda_hl_cig_p  := fcoalesce(bundle_nfda_hl_cig_actual,  bundle_hi_cig_cm, bundle_hi_cig_sm, bundle_hi_cig_m)]
dt_prices[, bundle_nfda_hl_ecig_p := fcoalesce(bundle_nfda_hl_ecig_actual, bundle_nfda_lo_ecig_cm, bundle_nfda_lo_ecig_sm, bundle_nfda_lo_ecig_m)]
dt_prices[, bundle_nfda_hh_cig_p  := fcoalesce(bundle_nfda_hh_cig_actual,  bundle_hi_cig_cm, bundle_hi_cig_sm, bundle_hi_cig_m)]
dt_prices[, bundle_nfda_hh_ecig_p := fcoalesce(bundle_nfda_hh_ecig_actual, bundle_nfda_hi_ecig_cm, bundle_nfda_hi_ecig_sm, bundle_nfda_hi_ecig_m)]
dt_prices[, bundle_fda_ll_cig_p   := fcoalesce(bundle_fda_ll_cig_actual,   bundle_lo_cig_cm, bundle_lo_cig_sm, bundle_lo_cig_m)]
dt_prices[, bundle_fda_ll_ecig_p  := fcoalesce(bundle_fda_ll_ecig_actual,  bundle_fda_lo_ecig_cm, bundle_fda_lo_ecig_sm, bundle_fda_lo_ecig_m)]
dt_prices[, bundle_fda_lh_cig_p   := fcoalesce(bundle_fda_lh_cig_actual,   bundle_lo_cig_cm, bundle_lo_cig_sm, bundle_lo_cig_m)]
dt_prices[, bundle_fda_lh_ecig_p  := fcoalesce(bundle_fda_lh_ecig_actual,  bundle_fda_hi_ecig_cm, bundle_fda_hi_ecig_sm, bundle_fda_hi_ecig_m)]
dt_prices[, bundle_fda_hl_cig_p   := fcoalesce(bundle_fda_hl_cig_actual,   bundle_hi_cig_cm, bundle_hi_cig_sm, bundle_hi_cig_m)]
dt_prices[, bundle_fda_hl_ecig_p  := fcoalesce(bundle_fda_hl_ecig_actual,  bundle_fda_lo_ecig_cm, bundle_fda_lo_ecig_sm, bundle_fda_lo_ecig_m)]
dt_prices[, bundle_fda_hh_cig_p   := fcoalesce(bundle_fda_hh_cig_actual,   bundle_hi_cig_cm, bundle_hi_cig_sm, bundle_hi_cig_m)]
dt_prices[, bundle_fda_hh_ecig_p  := fcoalesce(bundle_fda_hh_ecig_actual,  bundle_fda_hi_ecig_cm, bundle_fda_hi_ecig_sm, bundle_fda_hi_ecig_m)]

price_cols <- c(
  "cig_1_p", "cig_2_p", "cig_3to4_p", "cig_5to9_p", "cig_10_p", "cig_11to19_p",
  "cig_20_p", "cig_21to29_p", "cig_30_p", "cig_31to39_p", "cig_40_p", "cig_41plus_p",
  "ecig_0to5_p", "ecig_5to10_p", "ecig_10to15_p", "ecig_15to20_p",
  "ecig_20to30_p", "ecig_30to50_p", "ecig_50plus_p",
  "bundle_orig_ll_cig_p", "bundle_orig_lh_cig_p", "bundle_orig_hl_cig_p", "bundle_orig_hh_cig_p",
  "bundle_nfda_ll_cig_p", "bundle_nfda_lh_cig_p", "bundle_nfda_hl_cig_p", "bundle_nfda_hh_cig_p",
  "bundle_fda_ll_cig_p",  "bundle_fda_lh_cig_p",  "bundle_fda_hl_cig_p",  "bundle_fda_hh_cig_p",
  "bundle_orig_ll_ecig_p", "bundle_orig_lh_ecig_p", "bundle_orig_hl_ecig_p", "bundle_orig_hh_ecig_p",
  "bundle_nfda_ll_ecig_p", "bundle_nfda_lh_ecig_p", "bundle_nfda_hl_ecig_p", "bundle_nfda_hh_ecig_p",
  "bundle_fda_ll_ecig_p",  "bundle_fda_lh_ecig_p",  "bundle_fda_hl_ecig_p",  "bundle_fda_hh_ecig_p"
)

dt_prices[, sum(sapply(.SD, function(x) sum(is.na(x)))), .SDcols = price_cols]

dt_prices <- dt_prices[, ..price_cols]

file_name <- paste0(dir_holdout_val, "Prices.csv")
fwrite(dt_prices, file_name)

if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}

cat("\nData_Holdout_Val/ preparation complete.\n")
cat("Validation data: overlap households x 2023\n")
cat("Next step: update 02_Validation_Mixture.jl to load 2023 data from Data_Holdout_Val/\n")
cat("           and use terminal 2022 addiction stocks as 2023 starting states.\n")


#############################
# Household Coverage Check
#############################

# Verify that every 2023 overlap household appears in the 2021-2022 estimation
# sample. 
hh_est_codes <- fread(paste0(dir_holdout, "Household_Codes.csv"))[, unique(household_code)]
hh_val_codes <- fread(paste0(dir_holdout_val, "Household_Codes.csv"))[, unique(household_code)]
missing_from_est <- setdiff(hh_val_codes, hh_est_codes)
if (length(missing_from_est) == 0)
{
  cat("\nHousehold coverage check PASSED: all", length(hh_val_codes),
      "overlap households in Data_Holdout_Val/ are present in the Data_Holdout/ estimation sample.\n")
} else
{
  cat("\nWARNING: Household coverage check FAILED.", length(missing_from_est),
      "households in Data_Holdout_Val/ are NOT in the Data_Holdout/ estimation sample.\n")
}



