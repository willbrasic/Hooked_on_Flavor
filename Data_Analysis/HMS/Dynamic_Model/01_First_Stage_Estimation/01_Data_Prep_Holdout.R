################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script prepares holdout data for out-of-sample model evaluation.
#
# Estimation sample: ALL households x 2021-2022
# Validation sample: overlap households x 2023 
#
# Overlap households are those present in both 2021-2022 AND 2023. The full
# 2021-2022 sample is used for estimation so that parameters are not restricted
# to the overlap subset. The overlap household codes are saved separately so
# the validation script can identify which households have 2023 observations.
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
options(datatable.print.nrows = 200)

# Load in full panel data
file_name <- paste0("./all_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt_full <- fread(file_name)

# Output directory
dir_holdout <- "../Dynamic_Model/Data_Holdout/"
if (!dir.exists(dir_holdout))
{
  dir.create(dir_holdout)
}


#############################
# Identify Overlap Households
#############################

# purchase_month is a Date object; extract year with data.table's year()
dt_full[, year_pm := year(purchase_month)]

cat("Months in data:", format(sort(unique(dt_full[, purchase_month])), "%Y-%m"), "\n")

# Households observed in any 2021-2022 month (estimation window)
list_hh_est <- dt_full[year_pm %in% c(2021, 2022), unique(household_code)]

# Households observed in any 2023 month (validation window)
list_hh_val <- dt_full[year_pm == 2023, unique(household_code)]

# Overlap: present in both windows
list_overlap_hh <- intersect(list_hh_est, list_hh_val)
cat("Total households in 2021-2022:         ", length(list_hh_est),    "\n")
cat("Total households in 2023:              ", length(list_hh_val),    "\n")
cat("Overlap households (in both windows):  ", length(list_overlap_hh),"\n")

# Filter to ALL households x 2021-2022 for estimation
dt <- dt_full[year_pm %in% c(2021, 2022)]
dt[, year_pm := NULL]
dt_full[, year_pm := NULL]

cat("Estimation observations (all HHs x 2021-2022):", nrow(dt), "\n")

rm(dt_full, list_hh_est, list_hh_val)


#############################
# Top-Coding
#############################

# Monthly e-liquid quantities are top-coded at 100 mL, the lower edge of the open-ended bin
ECIG_TOP_CODE <- 100   # mL

dt[, total_original_mL_tc         := pmin(total_original_mL,         ECIG_TOP_CODE)]
dt[, total_non_fda_flavored_mL_tc := pmin(total_non_fda_flavored_mL, ECIG_TOP_CODE)]
dt[, total_fda_flavored_mL_tc     := pmin(total_fda_flavored_mL,     ECIG_TOP_CODE)]

# Nicotine absorbed is proportional to quantity, so it is capped in the same ratio
dt[, ecig_nicotine_mg_absorbed_tc := fifelse(total_mL > ECIG_TOP_CODE, ecig_nicotine_mg_absorbed * ECIG_TOP_CODE / total_mL, ecig_nicotine_mg_absorbed)]

cat("Top-coded e-liquid months above", ECIG_TOP_CODE, "mL:", dt[, sum(total_mL > ECIG_TOP_CODE)], "\n")


rm(list_overlap_hh)


#############################
# Household codes
#############################

# Household codes corresponding to each observation
dt_hh <- data.table("household_code" = dt[, household_code])

# Write the data to a file
file_name <- paste0(dir_holdout, "Household_Codes.csv")
fwrite(dt_hh, file_name)

# Confirm results have been written to a file
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

# Add in indicators for category
dt_category_choices <- dt[, .(
  outside_option,
  cig,
  ecig,
  cig_ecig
)]

# Check no NA values exist in the choice indicators
stopifnot("NA values found in category choice indicators" = !any(is.na(dt_category_choices)))

# Check all rows sum to 1 (exactly one category chosen per household-month)
stopifnot("Each household-month must have exactly one category choice" = all(rowSums(dt_category_choices) == 1))

# Write the data to a file
file_name <- paste0(dir_holdout, "Category_Choices.csv")
fwrite(dt_category_choices, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Bundle Split Thresholds
# (Median packs and mL by product type)
#############################

med_cig_packs <- dt[total_packs > 0 & total_mL == 0,          median(total_packs,               na.rm = TRUE)]
med_orig_mL   <- dt[total_original_mL > 0 & total_packs == 0, median(total_original_mL,          na.rm = TRUE)]
med_nfda_mL   <- dt[total_non_fda_flavored_mL > 0 & total_packs == 0, median(total_non_fda_flavored_mL, na.rm = TRUE)]
med_fda_mL    <- dt[total_fda_flavored_mL > 0 & total_packs == 0,     median(total_fda_flavored_mL,     na.rm = TRUE)]

cat("Bundle split thresholds:\n")
cat("  Cig packs median:          ", med_cig_packs, "\n")
cat("  Original ecig mL median:   ", med_orig_mL,   "\n")
cat("  Non-FDA flavored mL median:", med_nfda_mL,   "\n")
cat("  FDA flavored mL median:    ", med_fda_mL,    "\n")


#############################
# Product Level Choices
#############################

# Add indicators for product choices
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

  # Bundles: 4 per flavor type (lo/hi cig at median packs x lo/hi ecig at flavor-specific medians)
  # ll = lo cig + lo ecig, lh = lo cig + hi ecig, hl = hi cig + lo ecig, hh = hi cig + hi ecig
  # Original e-cig bundles
  bundle_orig_ll = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL, 1, 0),
  bundle_orig_lh = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL, 1, 0),
  bundle_orig_hl = fifelse(total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL, 1, 0),
  bundle_orig_hh = fifelse(total_packs > med_cig_packs & total_original_mL > med_orig_mL, 1, 0),
  # Non-FDA flavored bundles
  bundle_nfda_ll = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL, 1, 0),
  bundle_nfda_lh = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL, 1, 0),
  bundle_nfda_hl = fifelse(total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL, 1, 0),
  bundle_nfda_hh = fifelse(total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL, 1, 0),
  # FDA flavored bundles
  bundle_fda_ll  = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL, 1, 0),
  bundle_fda_lh  = fifelse(total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL, 1, 0),
  bundle_fda_hl  = fifelse(total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL, 1, 0),
  bundle_fda_hh  = fifelse(total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL, 1, 0)
)]

# Fraction of the time alternatives are chosen in all household-months
data.table(
  alternative = names(dt_product_choices),
  fraction = colMeans(dt_product_choices)
)[order(-fraction)]

# Check no NA values exist in the product choice indicators
stopifnot("NA values found in product choice indicators" = !any(is.na(dt_product_choices)))

# Check each household-month has exactly one chosen alternative
stopifnot("Each household-month must have exactly one product choice" = all(rowSums(dt_product_choices) == 1))

# Write the data to a file
file_name <- paste0(dir_holdout, "Product_Choices.csv")
fwrite(dt_product_choices, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Consumption
#############################

# Define consumption values for each alternative
dt_consumption <- dt[, .(

  # Cigarette alternatives (cigs only: total_mL == 0)
  cig_1      = 1,
  cig_2      = 2,
  cig_3to4   = median(total_packs[total_packs >= 3  & total_packs <= 4  & total_mL == 0], na.rm = TRUE),
  cig_5to9   = median(total_packs[total_packs >= 5  & total_packs <= 9  & total_mL == 0], na.rm = TRUE),
  cig_10     = 10,
  cig_11to19 = median(total_packs[total_packs >= 11 & total_packs <= 19 & total_mL == 0], na.rm = TRUE),
  cig_20     = 20,
  cig_21to29 = median(total_packs[total_packs >= 21 & total_packs <= 29 & total_mL == 0], na.rm = TRUE),
  cig_30     = 30,
  cig_31to39 = median(total_packs[total_packs >= 31 & total_packs <= 39 & total_mL == 0], na.rm = TRUE),
  cig_40     = 40,
  cig_41plus = median(total_packs[total_packs >= 41 & total_mL == 0], na.rm = TRUE),

  # Original e-cig alternatives (e-cig only: total_packs == 0)
  orig_ecig_0to5   = median(total_original_mL_tc[total_original_mL > 0  & total_original_mL <= 5  & total_packs == 0], na.rm = TRUE),
  orig_ecig_5to10  = median(total_original_mL_tc[total_original_mL > 5  & total_original_mL <= 10 & total_packs == 0], na.rm = TRUE),
  orig_ecig_10to15 = median(total_original_mL_tc[total_original_mL > 10 & total_original_mL <= 15 & total_packs == 0], na.rm = TRUE),
  orig_ecig_15to20 = median(total_original_mL_tc[total_original_mL > 15 & total_original_mL <= 20 & total_packs == 0], na.rm = TRUE),
  orig_ecig_20to30 = median(total_original_mL_tc[total_original_mL > 20 & total_original_mL <= 30 & total_packs == 0], na.rm = TRUE),
  orig_ecig_30to50 = median(total_original_mL_tc[total_original_mL > 30 & total_original_mL <= 50 & total_packs == 0], na.rm = TRUE),
  orig_ecig_50to100 = median(total_original_mL_tc[total_original_mL > 50 & total_original_mL <= 100 & total_packs == 0], na.rm = TRUE),
  orig_ecig_100plus = median(total_original_mL_tc[total_original_mL > 100 & total_packs == 0], na.rm = TRUE),

  # Non-FDA flavored e-cig alternatives (e-cig only: total_packs == 0)
  non_fda_flav_ecig_0to5   = median(total_non_fda_flavored_mL_tc[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= 5  & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_5to10  = median(total_non_fda_flavored_mL_tc[total_non_fda_flavored_mL > 5  & total_non_fda_flavored_mL <= 10 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_10to15 = median(total_non_fda_flavored_mL_tc[total_non_fda_flavored_mL > 10 & total_non_fda_flavored_mL <= 15 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_15to20 = median(total_non_fda_flavored_mL_tc[total_non_fda_flavored_mL > 15 & total_non_fda_flavored_mL <= 20 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_20to30 = median(total_non_fda_flavored_mL_tc[total_non_fda_flavored_mL > 20 & total_non_fda_flavored_mL <= 30 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_30to50 = median(total_non_fda_flavored_mL_tc[total_non_fda_flavored_mL > 30 & total_non_fda_flavored_mL <= 50 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_50to100 = median(total_non_fda_flavored_mL_tc[total_non_fda_flavored_mL > 50 & total_non_fda_flavored_mL <= 100 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_100plus = median(total_non_fda_flavored_mL_tc[total_non_fda_flavored_mL > 100 & total_packs == 0], na.rm = TRUE),

  # FDA-authorized flavored e-cig alternatives (e-cig only: total_packs == 0)
  fda_flav_ecig_0to5   = median(total_fda_flavored_mL_tc[total_fda_flavored_mL > 0  & total_fda_flavored_mL <= 5  & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_5to10  = median(total_fda_flavored_mL_tc[total_fda_flavored_mL > 5  & total_fda_flavored_mL <= 10 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_10to15 = median(total_fda_flavored_mL_tc[total_fda_flavored_mL > 10 & total_fda_flavored_mL <= 15 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_15to20 = median(total_fda_flavored_mL_tc[total_fda_flavored_mL > 15 & total_fda_flavored_mL <= 20 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_20to30 = median(total_fda_flavored_mL_tc[total_fda_flavored_mL > 20 & total_fda_flavored_mL <= 30 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_30to50 = median(total_fda_flavored_mL_tc[total_fda_flavored_mL > 30 & total_fda_flavored_mL <= 50 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_50to100 = median(total_fda_flavored_mL_tc[total_fda_flavored_mL > 50 & total_fda_flavored_mL <= 100 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_100plus = median(total_fda_flavored_mL_tc[total_fda_flavored_mL > 100 & total_packs == 0], na.rm = TRUE),

  # Bundle alternatives: 12 bundles (lo/hi cig x lo/hi ecig per flavor type)
  # Original e-cig bundles
  bundle_orig_ll_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_ll_ecig = median(total_original_mL_tc[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_lh_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_lh_ecig = median(total_original_mL_tc[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_hl_cig  = median(total_packs[total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_hl_ecig = median(total_original_mL_tc[total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_hh_cig  = median(total_packs[total_packs > med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_hh_ecig = median(total_original_mL_tc[total_packs > med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  # Non-FDA flavored e-cig bundles
  bundle_nfda_ll_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_ll_ecig = median(total_non_fda_flavored_mL_tc[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_lh_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_lh_ecig = median(total_non_fda_flavored_mL_tc[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hl_cig  = median(total_packs[total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hl_ecig = median(total_non_fda_flavored_mL_tc[total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hh_cig  = median(total_packs[total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hh_ecig = median(total_non_fda_flavored_mL_tc[total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  # FDA flavored e-cig bundles
  bundle_fda_ll_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_ll_ecig = median(total_fda_flavored_mL_tc[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_lh_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_lh_ecig = median(total_fda_flavored_mL_tc[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_hl_cig  = median(total_packs[total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_hl_ecig = median(total_fda_flavored_mL_tc[total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_hh_cig  = median(total_packs[total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_hh_ecig = median(total_fda_flavored_mL_tc[total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE)
)]

# Convert to long format
dt_consumption_long <- melt(
  dt_consumption,
  variable.name = "alternative",
  value.name    = "consumption"
)
dt_consumption_long[, alternative := as.character(alternative)]
dt_consumption_long

# Write the data to a file
file_name <- paste0(dir_holdout, "Consumption_Spaces.csv")
fwrite(dt_consumption_long, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Nicotine
#############################

# Define nicotine values for each alternative
dt_nicotine <- dt[, .(

  # Cigarette alternatives (cigs only: total_mL == 0)
  cig_1      = median(cig_nicotine_mg_absorbed[total_packs == 1  & total_mL == 0], na.rm = TRUE),
  cig_2      = median(cig_nicotine_mg_absorbed[total_packs == 2  & total_mL == 0], na.rm = TRUE),
  cig_3to4   = median(cig_nicotine_mg_absorbed[total_packs >= 3  & total_packs <= 4  & total_mL == 0], na.rm = TRUE),
  cig_5to9   = median(cig_nicotine_mg_absorbed[total_packs >= 5  & total_packs <= 9  & total_mL == 0], na.rm = TRUE),
  cig_10     = median(cig_nicotine_mg_absorbed[total_packs == 10 & total_mL == 0], na.rm = TRUE),
  cig_11to19 = median(cig_nicotine_mg_absorbed[total_packs >= 11 & total_packs <= 19 & total_mL == 0], na.rm = TRUE),
  cig_20     = median(cig_nicotine_mg_absorbed[total_packs == 20 & total_mL == 0], na.rm = TRUE),
  cig_21to29 = median(cig_nicotine_mg_absorbed[total_packs >= 21 & total_packs <= 29 & total_mL == 0], na.rm = TRUE),
  cig_30     = median(cig_nicotine_mg_absorbed[total_packs == 30 & total_mL == 0], na.rm = TRUE),
  cig_31to39 = median(cig_nicotine_mg_absorbed[total_packs >= 31 & total_packs <= 39 & total_mL == 0], na.rm = TRUE),
  cig_40     = median(cig_nicotine_mg_absorbed[total_packs == 40 & total_mL == 0], na.rm = TRUE),
  cig_41plus = median(cig_nicotine_mg_absorbed[total_packs >= 41 & total_mL == 0], na.rm = TRUE),

  # Original e-cig alternatives (e-cig only: total_packs == 0)
  orig_ecig_0to5   = median(ecig_nicotine_mg_absorbed_tc[total_original_mL > 0  & total_original_mL <= 5  & total_packs == 0], na.rm = TRUE),
  orig_ecig_5to10  = median(ecig_nicotine_mg_absorbed_tc[total_original_mL > 5  & total_original_mL <= 10 & total_packs == 0], na.rm = TRUE),
  orig_ecig_10to15 = median(ecig_nicotine_mg_absorbed_tc[total_original_mL > 10 & total_original_mL <= 15 & total_packs == 0], na.rm = TRUE),
  orig_ecig_15to20 = median(ecig_nicotine_mg_absorbed_tc[total_original_mL > 15 & total_original_mL <= 20 & total_packs == 0], na.rm = TRUE),
  orig_ecig_20to30 = median(ecig_nicotine_mg_absorbed_tc[total_original_mL > 20 & total_original_mL <= 30 & total_packs == 0], na.rm = TRUE),
  orig_ecig_30to50 = median(ecig_nicotine_mg_absorbed_tc[total_original_mL > 30 & total_original_mL <= 50 & total_packs == 0], na.rm = TRUE),
  orig_ecig_50to100 = median(ecig_nicotine_mg_absorbed_tc[total_original_mL > 50 & total_original_mL <= 100 & total_packs == 0], na.rm = TRUE),
  orig_ecig_100plus = median(ecig_nicotine_mg_absorbed_tc[total_original_mL > 100 & total_packs == 0], na.rm = TRUE),

  # Non-FDA flavored e-cig alternatives (e-cig only: total_packs == 0)
  non_fda_flav_ecig_0to5   = median(ecig_nicotine_mg_absorbed_tc[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= 5  & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_5to10  = median(ecig_nicotine_mg_absorbed_tc[total_non_fda_flavored_mL > 5  & total_non_fda_flavored_mL <= 10 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_10to15 = median(ecig_nicotine_mg_absorbed_tc[total_non_fda_flavored_mL > 10 & total_non_fda_flavored_mL <= 15 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_15to20 = median(ecig_nicotine_mg_absorbed_tc[total_non_fda_flavored_mL > 15 & total_non_fda_flavored_mL <= 20 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_20to30 = median(ecig_nicotine_mg_absorbed_tc[total_non_fda_flavored_mL > 20 & total_non_fda_flavored_mL <= 30 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_30to50 = median(ecig_nicotine_mg_absorbed_tc[total_non_fda_flavored_mL > 30 & total_non_fda_flavored_mL <= 50 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_50to100 = median(ecig_nicotine_mg_absorbed_tc[total_non_fda_flavored_mL > 50 & total_non_fda_flavored_mL <= 100 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_100plus = median(ecig_nicotine_mg_absorbed_tc[total_non_fda_flavored_mL > 100 & total_packs == 0], na.rm = TRUE),

  # FDA-authorized flavored e-cig alternatives (e-cig only: total_packs == 0)
  fda_flav_ecig_0to5   = median(ecig_nicotine_mg_absorbed_tc[total_fda_flavored_mL > 0  & total_fda_flavored_mL <= 5  & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_5to10  = median(ecig_nicotine_mg_absorbed_tc[total_fda_flavored_mL > 5  & total_fda_flavored_mL <= 10 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_10to15 = median(ecig_nicotine_mg_absorbed_tc[total_fda_flavored_mL > 10 & total_fda_flavored_mL <= 15 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_15to20 = median(ecig_nicotine_mg_absorbed_tc[total_fda_flavored_mL > 15 & total_fda_flavored_mL <= 20 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_20to30 = median(ecig_nicotine_mg_absorbed_tc[total_fda_flavored_mL > 20 & total_fda_flavored_mL <= 30 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_30to50 = median(ecig_nicotine_mg_absorbed_tc[total_fda_flavored_mL > 30 & total_fda_flavored_mL <= 50 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_50to100 = median(ecig_nicotine_mg_absorbed_tc[total_fda_flavored_mL > 50 & total_fda_flavored_mL <= 100 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_100plus = median(ecig_nicotine_mg_absorbed_tc[total_fda_flavored_mL > 100 & total_packs == 0], na.rm = TRUE),

  # Bundle alternatives: 12 bundles (lo/hi cig x lo/hi ecig per flavor type)
  # Original e-cig bundles
  bundle_orig_ll_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_ll_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_lh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_lh_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_hl_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_hl_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_hh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_hh_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  # Non-FDA flavored e-cig bundles
  bundle_nfda_ll_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_ll_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_lh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_lh_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hl_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hl_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hh_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  # FDA flavored e-cig bundles
  bundle_fda_ll_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_ll_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_lh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_lh_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_hl_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_hl_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_hh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_hh_ecig_nic = median(ecig_nicotine_mg_absorbed_tc[total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE)
)]

# Convert to long format
dt_nicotine_long <- melt(
  dt_nicotine,
  variable.name = "alternative",
  value.name    = "nicotine"
)
dt_nicotine_long[, alternative := as.character(alternative)]
dt_nicotine_long

# Write the data to a file
file_name <- paste0(dir_holdout, "Nicotine_Spaces.csv")
fwrite(dt_nicotine_long, file_name)

# Confirm results have been written to a file
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

# Data table indicating if youth or younger adult is present in household
dt_tya <- dt[, .(household_code, purchase_month, teen_or_young_adult_present)]

# Rename column
setnames(dt_tya, "teen_or_young_adult_present", "tya_state")

# Write the data to a file
file_name <- paste0(dir_holdout, "TYA_States.csv")
fwrite(dt_tya, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Price Ratios
# (Quantity Discounts)
#############################

# Overall category medians (across all purchasers regardless of quantity bin)
overall_cig_price  <- dt[total_packs > 0, median(real_per_pack_price_paid, na.rm = TRUE)]
overall_ecig_price <- dt[total_mL > 0,    median(real_per_mL_price_paid, na.rm = TRUE)]

# Cigarette bin medians and ratios
dt_price_ratios <- data.table(
  alternative = c(
    # Cigarette bins
    "cig_1", "cig_2", "cig_3to4", "cig_5to9", "cig_10", "cig_11to19",
    "cig_20", "cig_21to29", "cig_30", "cig_31to39", "cig_40", "cig_41plus",
    # E-cig bins (pooled orig/flav, by total mL)
    "ecig_0to5", "ecig_5to10", "ecig_10to15", "ecig_15to20",
    "ecig_20to30", "ecig_30to50", "ecig_50to100", "ecig_100plus"
  ),
  median_price = c(
    # Cigarette bin medians
    dt[total_packs == 1,                              median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs == 2,                              median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs >= 3  & total_packs <= 4,          median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs >= 5  & total_packs <= 9,          median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs == 10,                             median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs >= 11 & total_packs <= 19,         median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs == 20,                             median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs >= 21 & total_packs <= 29,         median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs == 30,                             median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs >= 31 & total_packs <= 39,         median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs == 40,                             median(real_per_pack_price_paid, na.rm = TRUE)],
    dt[total_packs >= 41,                             median(real_per_pack_price_paid, na.rm = TRUE)],
    # E-cig bin medians (pooled orig/flav, by total mL)
    dt[total_mL > 0  & total_mL <= 5,  median(real_per_mL_price_paid, na.rm = TRUE)],
    dt[total_mL > 5  & total_mL <= 10, median(real_per_mL_price_paid, na.rm = TRUE)],
    dt[total_mL > 10 & total_mL <= 15, median(real_per_mL_price_paid, na.rm = TRUE)],
    dt[total_mL > 15 & total_mL <= 20, median(real_per_mL_price_paid, na.rm = TRUE)],
    dt[total_mL > 20 & total_mL <= 30, median(real_per_mL_price_paid, na.rm = TRUE)],
    dt[total_mL > 30 & total_mL <= 50, median(real_per_mL_price_paid, na.rm = TRUE)],
    dt[total_mL > 50 & total_mL <= 100, median(real_per_mL_price_paid, na.rm = TRUE)],

    # 100+ bin: quantity is top-coded at ECIG_TOP_CODE, so price it per modeled mL
    # (spend / ECIG_TOP_CODE) rather than per mL actually bought
    dt[total_mL > 100, median(real_per_mL_price_paid * total_mL / ECIG_TOP_CODE, na.rm = TRUE)]
  ),
  overall_median = c(
    rep(overall_cig_price, 12),
    rep(overall_ecig_price, 8)
  )
)

# Compute ratios
dt_price_ratios[, ratio := median_price / overall_median]
dt_price_ratios

# Write to file
file_name <- paste0(dir_holdout, "Price_Ratios.csv")
fwrite(dt_price_ratios, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("\nResults have been written to\n", file_name, "\n")
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
file_name <- paste0(dir_holdout, "Prices.csv")
fwrite(dt_prices, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}

cat("\nData_Holdout/ preparation complete.\n")
cat("Estimation data: ALL households x 2021-2022\n")
