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
# Consumption_Spaces, Nicotine_Spaces, Pricing_Spaces, Price_Ratios, and
# Halton_Draw_Transitions are copied from Data_Holdout/ unchanged.
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
  "Consumption_Spaces.csv",
  "Nicotine_Spaces.csv",
  "Pricing_Spaces.csv",
  "Halton_Draw_Transitions.csv",
  "Price_Ratios.csv"
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
stopifnot("Each household-month must have exactly one category choice" = all(rowSums(dt_category_choices) == 1))

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
  orig_ecig_50to100 = fifelse(total_original_mL > 50 & total_original_mL <= 100 & total_packs == 0, 1, 0),
  orig_ecig_100plus = fifelse(total_original_mL > 100 & total_packs == 0, 1, 0),

  # Non-FDA flavored e-cigarette alternatives (e-cig only: total_packs == 0)
  non_fda_flav_ecig_0to5    = fifelse(total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= 5  & total_packs == 0, 1, 0),
  non_fda_flav_ecig_5to10   = fifelse(total_non_fda_flavored_mL > 5  & total_non_fda_flavored_mL <= 10 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_10to15  = fifelse(total_non_fda_flavored_mL > 10 & total_non_fda_flavored_mL <= 15 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_15to20  = fifelse(total_non_fda_flavored_mL > 15 & total_non_fda_flavored_mL <= 20 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_20to30  = fifelse(total_non_fda_flavored_mL > 20 & total_non_fda_flavored_mL <= 30 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_30to50  = fifelse(total_non_fda_flavored_mL > 30 & total_non_fda_flavored_mL <= 50 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_50to100 = fifelse(total_non_fda_flavored_mL > 50 & total_non_fda_flavored_mL <= 100 & total_packs == 0, 1, 0),
  non_fda_flav_ecig_100plus = fifelse(total_non_fda_flavored_mL > 100 & total_packs == 0, 1, 0),

  # FDA-authorized flavored e-cigarette alternatives (e-cig only: total_packs == 0)
  fda_flav_ecig_0to5    = fifelse(total_fda_flavored_mL > 0  & total_fda_flavored_mL <= 5  & total_packs == 0, 1, 0),
  fda_flav_ecig_5to10   = fifelse(total_fda_flavored_mL > 5  & total_fda_flavored_mL <= 10 & total_packs == 0, 1, 0),
  fda_flav_ecig_10to15  = fifelse(total_fda_flavored_mL > 10 & total_fda_flavored_mL <= 15 & total_packs == 0, 1, 0),
  fda_flav_ecig_15to20  = fifelse(total_fda_flavored_mL > 15 & total_fda_flavored_mL <= 20 & total_packs == 0, 1, 0),
  fda_flav_ecig_20to30  = fifelse(total_fda_flavored_mL > 20 & total_fda_flavored_mL <= 30 & total_packs == 0, 1, 0),
  fda_flav_ecig_30to50  = fifelse(total_fda_flavored_mL > 30 & total_fda_flavored_mL <= 50 & total_packs == 0, 1, 0),
  fda_flav_ecig_50to100 = fifelse(total_fda_flavored_mL > 50 & total_fda_flavored_mL <= 100 & total_packs == 0, 1, 0),
  fda_flav_ecig_100plus = fifelse(total_fda_flavored_mL > 100 & total_packs == 0, 1, 0),

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
stopifnot("Each household-month must have exactly one product choice" = all(rowSums(dt_product_choices) == 1))

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
# (Market price x bin ratio)
#############################

# One market price per household-month and category: the county-month median
# per-unit price pooled across quantity bins when the county has at least
# `min_cell` purchases of that category in the month, otherwise the state-month
# median, otherwise the national monthly median. A household's own transaction
# price enters only through these medians, so the alternative it chose is priced
# the same way as every alternative it did not choose. Bin-specific prices are
# the market price times the national bin price ratio, which carries the
# quantity discount. Estimation and simulation use this same construction.
# Bin price ratios and representative consumption come from the training sample
dt_price_ratios     <- fread(paste0(dir_holdout, "Price_Ratios.csv"))
dt_consumption_long <- fread(paste0(dir_holdout, "Consumption_Spaces.csv"))

min_cell <- 3

# County-month market prices (pooled across bins) and the number of purchases behind them
dt_county_month_prices <- dt[, .(
  cig_cm    = median(real_per_pack_price_paid[total_packs > 0], na.rm = TRUE),
  n_cig_cm  = sum(total_packs > 0),
  ecig_cm   = median(real_per_mL_price_paid[total_mL > 0], na.rm = TRUE),
  n_ecig_cm = sum(total_mL > 0)
), keyby = .(fips_state_code, fips_county_code, purchase_month)]

# State-month market prices
dt_state_month_prices <- dt[, .(
  cig_sm  = median(real_per_pack_price_paid[total_packs > 0], na.rm = TRUE),
  ecig_sm = median(real_per_mL_price_paid[total_mL > 0], na.rm = TRUE)
), keyby = .(fips_state_code, purchase_month)]

# National monthly market prices (last fallback)
dt_month_prices <- dt[, .(
  cig_m  = median(real_per_pack_price_paid[total_packs > 0], na.rm = TRUE),
  ecig_m = median(real_per_mL_price_paid[total_mL > 0], na.rm = TRUE)
), keyby = purchase_month]
stopifnot("Every month needs a national median price for both categories" = nrow(dt_month_prices[!complete.cases(dt_month_prices)]) == 0)

# Link market prices to every household-month
dt_prices <- dt[, .(household_code, purchase_month, fips_state_code, fips_county_code)]
dt_prices[dt_county_month_prices, on = .(fips_state_code, fips_county_code, purchase_month),
          `:=` (cig_cm = i.cig_cm, n_cig_cm = i.n_cig_cm, ecig_cm = i.ecig_cm, n_ecig_cm = i.n_ecig_cm)]
dt_prices[dt_state_month_prices, on = .(fips_state_code, purchase_month),
          `:=` (cig_sm = i.cig_sm, ecig_sm = i.ecig_sm)]
dt_prices[dt_month_prices, on = .(purchase_month),
          `:=` (cig_m = i.cig_m, ecig_m = i.ecig_m)]

# Market price: county-month if the cell is large enough, else state-month, else national
dt_prices[, market_cig  := fcoalesce(fifelse(!is.na(n_cig_cm)  & n_cig_cm  >= min_cell, cig_cm,  NA_real_), cig_sm,  cig_m)]
dt_prices[, market_ecig := fcoalesce(fifelse(!is.na(n_ecig_cm) & n_ecig_cm >= min_cell, ecig_cm, NA_real_), ecig_sm, ecig_m)]

# Share of household-months whose market price comes from each level
cat("Market price source (share of household-months):\n")
print(dt_prices[, .(
  cig_county  = mean(!is.na(n_cig_cm) & n_cig_cm >= min_cell),
  cig_state   = mean((is.na(n_cig_cm) | n_cig_cm < min_cell) & !is.na(cig_sm)),
  cig_nation  = mean((is.na(n_cig_cm) | n_cig_cm < min_cell) & is.na(cig_sm)),
  ecig_county = mean(!is.na(n_ecig_cm) & n_ecig_cm >= min_cell),
  ecig_state  = mean((is.na(n_ecig_cm) | n_ecig_cm < min_cell) & !is.na(ecig_sm)),
  ecig_nation = mean((is.na(n_ecig_cm) | n_ecig_cm < min_cell) & is.na(ecig_sm))
)])

# Bin price ratios (bin median / overall median) and representative consumption per alternative
ratio_lookup <- setNames(dt_price_ratios[, ratio], dt_price_ratios[, alternative])
cons_lookup  <- setNames(dt_consumption_long[, consumption], dt_consumption_long[, alternative])
cig_bins  <- c("cig_1", "cig_2", "cig_3to4", "cig_5to9", "cig_10", "cig_11to19",
               "cig_20", "cig_21to29", "cig_30", "cig_31to39", "cig_40", "cig_41plus")
ecig_bins <- c("ecig_0to5", "ecig_5to10", "ecig_10to15", "ecig_15to20",
               "ecig_20to30", "ecig_30to50", "ecig_50to100", "ecig_100plus")
ecig_cons_bins <- paste0("orig_", ecig_bins)   # consumption of the pooled e-cig bins (same for all flavor types)

# Standalone alternatives: market price x bin ratio
for (b in cig_bins)
{
  dt_prices[, (paste0(b, "_p")) := market_cig * ratio_lookup[[b]]]
}
for (b in ecig_bins)
{
  dt_prices[, (paste0(b, "_p")) := market_ecig * ratio_lookup[[b]]]
}

# Bundle alternatives: each component takes the ratio of the standalone bin whose
# representative consumption is closest to the bundle component's representative
# consumption (the same mapping the Julia code uses for bundles)
nearest_ratio <- function(value, cons_names, ratio_names)
{
  ratio_lookup[[ratio_names[which.min(abs(cons_lookup[cons_names] - value))]]]
}
bundles <- c("bundle_orig_ll", "bundle_orig_lh", "bundle_orig_hl", "bundle_orig_hh",
             "bundle_nfda_ll", "bundle_nfda_lh", "bundle_nfda_hl", "bundle_nfda_hh",
             "bundle_fda_ll",  "bundle_fda_lh",  "bundle_fda_hl",  "bundle_fda_hh")
for (bn in bundles)
{
  r_cig  <- nearest_ratio(cons_lookup[[paste0(bn, "_cig")]],  cig_bins,       cig_bins)
  r_ecig <- nearest_ratio(cons_lookup[[paste0(bn, "_ecig")]], ecig_cons_bins, ecig_bins)
  dt_prices[, (paste0(bn, "_cig_p"))  := market_cig  * r_cig]
  dt_prices[, (paste0(bn, "_ecig_p")) := market_ecig * r_ecig]
}

# Columns written: the 45 alternative prices (names unchanged for the Julia reader)
# plus the two market prices they are built from
price_cols <- c(
  paste0(cig_bins, "_p"),
  paste0(ecig_bins, "_p"),
  paste0(bundles, "_cig_p"),
  paste0(bundles, "_ecig_p"),
  "market_cig", "market_ecig"
)
stopifnot("Price matrix contains NA" = dt_prices[, sum(sapply(.SD, function(x) sum(is.na(x)))), .SDcols = price_cols] == 0)
dt_prices <- dt_prices[, ..price_cols]

# Write the data to a file
file_name <- paste0(dir_holdout_val, "Prices.csv")
fwrite(dt_prices, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}
#############################
# Household Coverage Check
#############################

# Check that every 2023 overlap household appears in the 2021-2022 estimation
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



