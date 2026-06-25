################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script prepares data to estimate the dynamic demand
# model
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
dt <- fread(file_name)



#############################
# Household codes 
############################# 

# Household codes corresponding to each observation
dt_hh <- data.table("household_code" = dt[, household_code])

# Write the data to a file
file_name <- "../Dynamic_Model/Data/Household_Codes.csv"
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
all(rowSums(dt_category_choices) == 1)

# Write the data to a file
file_name <- "../Dynamic_Model/Data/Category_Choices.csv"
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
# 
# # Summary statistics for bundle purchasers
# dt_bundles <- dt[cig == 1 & ecig == 1]
# summary_packs <- dt_bundles[
#   , .(
#     mean   = mean(total_packs, na.rm = TRUE),
#     median = median(total_packs, na.rm = TRUE),
#     min    = min(total_packs, na.rm = TRUE),
#     max    = max(total_packs, na.rm = TRUE),
#     p25    = quantile(total_packs, 0.25, na.rm = TRUE),
#     p50    = quantile(total_packs, 0.50, na.rm = TRUE),
#     p75    = quantile(total_packs, 0.75, na.rm = TRUE)
#   )
# ]
# summary_orig_ml <- dt_bundles[
#   total_original_mL > 0,
#   .(
#     mean   = mean(total_original_mL, na.rm = TRUE),
#     median = median(total_original_mL, na.rm = TRUE),
#     min    = min(total_original_mL, na.rm = TRUE),
#     max    = max(total_original_mL, na.rm = TRUE),
#     p25    = quantile(total_original_mL, 0.25, na.rm = TRUE),
#     p50    = quantile(total_original_mL, 0.50, na.rm = TRUE),
#     p75    = quantile(total_original_mL, 0.75, na.rm = TRUE)
#   )
# ]
# summary_flav_ml <- dt_bundles[
#   total_flavored_mL > 0,
#   .(
#     mean   = mean(total_flavored_mL, na.rm = TRUE),
#     median = median(total_flavored_mL, na.rm = TRUE),
#     min    = min(total_flavored_mL, na.rm = TRUE),
#     max    = max(total_flavored_mL, na.rm = TRUE),
#     p25    = quantile(total_flavored_mL, 0.25, na.rm = TRUE),
#     p50    = quantile(total_flavored_mL, 0.50, na.rm = TRUE),
#     p75    = quantile(total_flavored_mL, 0.75, na.rm = TRUE)
#   )
# ]

# Fraction of the time alternatives are chosen in all household-months
data.table(
  alternative = names(dt_product_choices),
  fraction = colMeans(dt_product_choices)
)[order(-fraction)]

# Check no NA values exist in the product choice indicators
stopifnot("NA values found in product choice indicators" = !any(is.na(dt_product_choices)))

# Ensure each household-month contains one chosen alternative
all(rowSums(dt_product_choices) == 1)

# Write the data to a file
file_name <- "../Dynamic_Model/Data/Product_Choices.csv"
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
  orig_ecig_0to5   = median(total_original_mL[total_original_mL > 0  & total_original_mL <= 5  & total_packs == 0], na.rm = TRUE),
  orig_ecig_5to10  = median(total_original_mL[total_original_mL > 5  & total_original_mL <= 10 & total_packs == 0], na.rm = TRUE),
  orig_ecig_10to15 = median(total_original_mL[total_original_mL > 10 & total_original_mL <= 15 & total_packs == 0], na.rm = TRUE),
  orig_ecig_15to20 = median(total_original_mL[total_original_mL > 15 & total_original_mL <= 20 & total_packs == 0], na.rm = TRUE),
  orig_ecig_20to30 = median(total_original_mL[total_original_mL > 20 & total_original_mL <= 30 & total_packs == 0], na.rm = TRUE),
  orig_ecig_30to50 = median(total_original_mL[total_original_mL > 30 & total_original_mL <= 50 & total_packs == 0], na.rm = TRUE),
  orig_ecig_50plus = median(total_original_mL[total_original_mL > 50 & total_packs == 0], na.rm = TRUE),

  # Non-FDA flavored e-cig alternatives (e-cig only: total_packs == 0)
  non_fda_flav_ecig_0to5   = median(total_non_fda_flavored_mL[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= 5  & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_5to10  = median(total_non_fda_flavored_mL[total_non_fda_flavored_mL > 5  & total_non_fda_flavored_mL <= 10 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_10to15 = median(total_non_fda_flavored_mL[total_non_fda_flavored_mL > 10 & total_non_fda_flavored_mL <= 15 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_15to20 = median(total_non_fda_flavored_mL[total_non_fda_flavored_mL > 15 & total_non_fda_flavored_mL <= 20 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_20to30 = median(total_non_fda_flavored_mL[total_non_fda_flavored_mL > 20 & total_non_fda_flavored_mL <= 30 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_30to50 = median(total_non_fda_flavored_mL[total_non_fda_flavored_mL > 30 & total_non_fda_flavored_mL <= 50 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_50plus = median(total_non_fda_flavored_mL[total_non_fda_flavored_mL > 50 & total_packs == 0], na.rm = TRUE),

  # FDA-authorized flavored e-cig alternatives (e-cig only: total_packs == 0)
  fda_flav_ecig_0to5   = median(total_fda_flavored_mL[total_fda_flavored_mL > 0  & total_fda_flavored_mL <= 5  & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_5to10  = median(total_fda_flavored_mL[total_fda_flavored_mL > 5  & total_fda_flavored_mL <= 10 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_10to15 = median(total_fda_flavored_mL[total_fda_flavored_mL > 10 & total_fda_flavored_mL <= 15 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_15to20 = median(total_fda_flavored_mL[total_fda_flavored_mL > 15 & total_fda_flavored_mL <= 20 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_20to30 = median(total_fda_flavored_mL[total_fda_flavored_mL > 20 & total_fda_flavored_mL <= 30 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_30to50 = median(total_fda_flavored_mL[total_fda_flavored_mL > 30 & total_fda_flavored_mL <= 50 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_50plus = median(total_fda_flavored_mL[total_fda_flavored_mL > 50 & total_packs == 0], na.rm = TRUE),

  # Bundle alternatives: 12 bundles (lo/hi cig x lo/hi ecig per flavor type)
  # Original e-cig bundles
  bundle_orig_ll_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_ll_ecig = median(total_original_mL[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_lh_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_lh_ecig = median(total_original_mL[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_hl_cig  = median(total_packs[total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_hl_ecig = median(total_original_mL[total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_hh_cig  = median(total_packs[total_packs > med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_hh_ecig = median(total_original_mL[total_packs > med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  # Non-FDA flavored e-cig bundles
  bundle_nfda_ll_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_ll_ecig = median(total_non_fda_flavored_mL[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_lh_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_lh_ecig = median(total_non_fda_flavored_mL[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hl_cig  = median(total_packs[total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hl_ecig = median(total_non_fda_flavored_mL[total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hh_cig  = median(total_packs[total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hh_ecig = median(total_non_fda_flavored_mL[total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  # FDA flavored e-cig bundles
  bundle_fda_ll_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_ll_ecig = median(total_fda_flavored_mL[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_lh_cig  = median(total_packs[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_lh_ecig = median(total_fda_flavored_mL[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_hl_cig  = median(total_packs[total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_hl_ecig = median(total_fda_flavored_mL[total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_hh_cig  = median(total_packs[total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_hh_ecig = median(total_fda_flavored_mL[total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE)
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
file_name <- "../Dynamic_Model/Data/Consumption_Spaces.csv"
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
  orig_ecig_0to5   = median(ecig_nicotine_mg_absorbed[total_original_mL > 0  & total_original_mL <= 5  & total_packs == 0], na.rm = TRUE),
  orig_ecig_5to10  = median(ecig_nicotine_mg_absorbed[total_original_mL > 5  & total_original_mL <= 10 & total_packs == 0], na.rm = TRUE),
  orig_ecig_10to15 = median(ecig_nicotine_mg_absorbed[total_original_mL > 10 & total_original_mL <= 15 & total_packs == 0], na.rm = TRUE),
  orig_ecig_15to20 = median(ecig_nicotine_mg_absorbed[total_original_mL > 15 & total_original_mL <= 20 & total_packs == 0], na.rm = TRUE),
  orig_ecig_20to30 = median(ecig_nicotine_mg_absorbed[total_original_mL > 20 & total_original_mL <= 30 & total_packs == 0], na.rm = TRUE),
  orig_ecig_30to50 = median(ecig_nicotine_mg_absorbed[total_original_mL > 30 & total_original_mL <= 50 & total_packs == 0], na.rm = TRUE),
  orig_ecig_50plus = median(ecig_nicotine_mg_absorbed[total_original_mL > 50 & total_packs == 0], na.rm = TRUE),

  # Non-FDA flavored e-cig alternatives (e-cig only: total_packs == 0)
  non_fda_flav_ecig_0to5   = median(ecig_nicotine_mg_absorbed[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= 5  & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_5to10  = median(ecig_nicotine_mg_absorbed[total_non_fda_flavored_mL > 5  & total_non_fda_flavored_mL <= 10 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_10to15 = median(ecig_nicotine_mg_absorbed[total_non_fda_flavored_mL > 10 & total_non_fda_flavored_mL <= 15 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_15to20 = median(ecig_nicotine_mg_absorbed[total_non_fda_flavored_mL > 15 & total_non_fda_flavored_mL <= 20 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_20to30 = median(ecig_nicotine_mg_absorbed[total_non_fda_flavored_mL > 20 & total_non_fda_flavored_mL <= 30 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_30to50 = median(ecig_nicotine_mg_absorbed[total_non_fda_flavored_mL > 30 & total_non_fda_flavored_mL <= 50 & total_packs == 0], na.rm = TRUE),
  non_fda_flav_ecig_50plus = median(ecig_nicotine_mg_absorbed[total_non_fda_flavored_mL > 50 & total_packs == 0], na.rm = TRUE),

  # FDA-authorized flavored e-cig alternatives (e-cig only: total_packs == 0)
  fda_flav_ecig_0to5   = median(ecig_nicotine_mg_absorbed[total_fda_flavored_mL > 0  & total_fda_flavored_mL <= 5  & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_5to10  = median(ecig_nicotine_mg_absorbed[total_fda_flavored_mL > 5  & total_fda_flavored_mL <= 10 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_10to15 = median(ecig_nicotine_mg_absorbed[total_fda_flavored_mL > 10 & total_fda_flavored_mL <= 15 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_15to20 = median(ecig_nicotine_mg_absorbed[total_fda_flavored_mL > 15 & total_fda_flavored_mL <= 20 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_20to30 = median(ecig_nicotine_mg_absorbed[total_fda_flavored_mL > 20 & total_fda_flavored_mL <= 30 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_30to50 = median(ecig_nicotine_mg_absorbed[total_fda_flavored_mL > 30 & total_fda_flavored_mL <= 50 & total_packs == 0], na.rm = TRUE),
  fda_flav_ecig_50plus = median(ecig_nicotine_mg_absorbed[total_fda_flavored_mL > 50 & total_packs == 0], na.rm = TRUE),

  # Bundle alternatives: 12 bundles (lo/hi cig x lo/hi ecig per flavor type)
  # Original e-cig bundles
  bundle_orig_ll_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_ll_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_lh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_lh_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_hl_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_hl_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_original_mL > 0  & total_original_mL <= med_orig_mL], na.rm = TRUE),
  bundle_orig_hh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  bundle_orig_hh_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_original_mL > med_orig_mL], na.rm = TRUE),
  # Non-FDA flavored e-cig bundles
  bundle_nfda_ll_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_ll_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_lh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_lh_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hl_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hl_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  bundle_nfda_hh_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_non_fda_flavored_mL > med_nfda_mL], na.rm = TRUE),
  # FDA flavored e-cig bundles
  bundle_fda_ll_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_ll_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_lh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_lh_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > 0 & total_packs <= med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_hl_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_hl_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_fda_flavored_mL > 0  & total_fda_flavored_mL <= med_fda_mL], na.rm = TRUE),
  bundle_fda_hh_cig_nic  = median(cig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE),
  bundle_fda_hh_ecig_nic = median(ecig_nicotine_mg_absorbed[total_packs > med_cig_packs & total_fda_flavored_mL > med_fda_mL], na.rm = TRUE)
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
file_name <- "../Dynamic_Model/Data/Nicotine_Spaces.csv"
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

# Number of unique households that change their teen or young adult status over time
dt_tya[, .(n_unique = uniqueN(teen_or_young_adult_present, na.rm = TRUE)), by = household_code][n_unique > 1, .N]

# Household codes that change their teen or young adult status across years
households_changing <- dt_tya[
  , .(n_unique = uniqueN(teen_or_young_adult_present, na.rm = TRUE)),
  by = household_code
][n_unique > 1, household_code]

# Get those observations for which teen or young adult status changes
dt_tya[household_code %in% households_changing]

# Rename column
setnames(dt_tya, "teen_or_young_adult_present", "tya_state")

# Write the data to a file
file_name <- "../Dynamic_Model/Data/TYA_States.csv"
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
# Prices
#############################


# County-month median per-unit prices BY BIN
# Adds within-state price variation
dt_county_month_prices <- dt[, .(
  
  # Cigarette price bins (by pack quantity, all purchasers)
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
  
  # E-cigarette price bins (per-mL price by total mL, pooling orig and flav)
  ecig_0to5_cm   = median(real_per_mL_price_paid[total_mL > 0  & total_mL <= 5],  na.rm = TRUE),
  ecig_5to10_cm  = median(real_per_mL_price_paid[total_mL > 5  & total_mL <= 10], na.rm = TRUE),
  ecig_10to15_cm = median(real_per_mL_price_paid[total_mL > 10 & total_mL <= 15], na.rm = TRUE),
  ecig_15to20_cm = median(real_per_mL_price_paid[total_mL > 15 & total_mL <= 20], na.rm = TRUE),
  ecig_20to30_cm = median(real_per_mL_price_paid[total_mL > 20 & total_mL <= 30], na.rm = TRUE),
  ecig_30to50_cm = median(real_per_mL_price_paid[total_mL > 30 & total_mL <= 50], na.rm = TRUE),
  ecig_50plus_cm = median(real_per_mL_price_paid[total_mL > 50], na.rm = TRUE),

  # Bundle price bins: cig split at med_cig_packs; ecig split by flavor type at flavor-specific medians
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
# Larger quantity bins may have lower per-unit prices (quantity discounts)
# Computed for all observations regardless of choice (no conditioning on outcome)
dt_state_month_prices <- dt[, .(

  # Cigarette price bins (by pack quantity, all purchasers)
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

  # E-cigarette price bins (per-mL price by total mL, pooling orig and flav)
  ecig_0to5_sm   = median(real_per_mL_price_paid[total_mL > 0  & total_mL <= 5],  na.rm = TRUE),
  ecig_5to10_sm  = median(real_per_mL_price_paid[total_mL > 5  & total_mL <= 10], na.rm = TRUE),
  ecig_10to15_sm = median(real_per_mL_price_paid[total_mL > 10 & total_mL <= 15], na.rm = TRUE),
  ecig_15to20_sm = median(real_per_mL_price_paid[total_mL > 15 & total_mL <= 20], na.rm = TRUE),
  ecig_20to30_sm = median(real_per_mL_price_paid[total_mL > 20 & total_mL <= 30], na.rm = TRUE),
  ecig_30to50_sm = median(real_per_mL_price_paid[total_mL > 30 & total_mL <= 50], na.rm = TRUE),
  ecig_50plus_sm = median(real_per_mL_price_paid[total_mL > 50], na.rm = TRUE),

  # Bundle price bins: cig split at med_cig_packs; ecig split by flavor type at flavor-specific medians
  bundle_lo_cig_sm       = median(real_per_pack_price_paid[total_packs > 0  & total_packs <= med_cig_packs], na.rm = TRUE),
  bundle_hi_cig_sm       = median(real_per_pack_price_paid[total_packs > med_cig_packs],                     na.rm = TRUE),
  bundle_orig_lo_ecig_sm = median(real_per_mL_price_paid[total_original_mL > 0       & total_original_mL <= med_orig_mL & total_packs == 0], na.rm = TRUE),
  bundle_orig_hi_ecig_sm = median(real_per_mL_price_paid[total_original_mL > med_orig_mL & total_packs == 0],                                na.rm = TRUE),
  bundle_nfda_lo_ecig_sm = median(real_per_mL_price_paid[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL & total_packs == 0], na.rm = TRUE),
  bundle_nfda_hi_ecig_sm = median(real_per_mL_price_paid[total_non_fda_flavored_mL > med_nfda_mL & total_packs == 0],                                    na.rm = TRUE),
  bundle_fda_lo_ecig_sm  = median(real_per_mL_price_paid[total_fda_flavored_mL > 0   & total_fda_flavored_mL <= med_fda_mL & total_packs == 0],          na.rm = TRUE),
  bundle_fda_hi_ecig_sm  = median(real_per_mL_price_paid[total_fda_flavored_mL > med_fda_mL & total_packs == 0],                                         na.rm = TRUE)

), keyby = .(fips_state_code, purchase_month)]

# Monthly median per-unit prices BY BIN (fallback when county-month and state-month medians are unavailable)
dt_month_prices <- dt[, .(

  # Cigarette price bins (by pack quantity, all purchasers)
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

  # E-cigarette price bins (per-mL price by total mL, pooling orig and flav)
  ecig_0to5_m   = median(real_per_mL_price_paid[total_mL > 0  & total_mL <= 5],  na.rm = TRUE),
  ecig_5to10_m  = median(real_per_mL_price_paid[total_mL > 5  & total_mL <= 10], na.rm = TRUE),
  ecig_10to15_m = median(real_per_mL_price_paid[total_mL > 10 & total_mL <= 15], na.rm = TRUE),
  ecig_15to20_m = median(real_per_mL_price_paid[total_mL > 15 & total_mL <= 20], na.rm = TRUE),
  ecig_20to30_m = median(real_per_mL_price_paid[total_mL > 20 & total_mL <= 30], na.rm = TRUE),
  ecig_30to50_m = median(real_per_mL_price_paid[total_mL > 30 & total_mL <= 50], na.rm = TRUE),
  ecig_50plus_m = median(real_per_mL_price_paid[total_mL > 50], na.rm = TRUE),

  # Bundle price bins: cig split at med_cig_packs; ecig split by flavor type at flavor-specific medians
  bundle_lo_cig_m       = median(real_per_pack_price_paid[total_packs > 0  & total_packs <= med_cig_packs], na.rm = TRUE),
  bundle_hi_cig_m       = median(real_per_pack_price_paid[total_packs > med_cig_packs],                     na.rm = TRUE),
  bundle_orig_lo_ecig_m = median(real_per_mL_price_paid[total_original_mL > 0       & total_original_mL <= med_orig_mL & total_packs == 0], na.rm = TRUE),
  bundle_orig_hi_ecig_m = median(real_per_mL_price_paid[total_original_mL > med_orig_mL & total_packs == 0],                                na.rm = TRUE),
  bundle_nfda_lo_ecig_m = median(real_per_mL_price_paid[total_non_fda_flavored_mL > 0  & total_non_fda_flavored_mL <= med_nfda_mL & total_packs == 0], na.rm = TRUE),
  bundle_nfda_hi_ecig_m = median(real_per_mL_price_paid[total_non_fda_flavored_mL > med_nfda_mL & total_packs == 0],                                    na.rm = TRUE),
  bundle_fda_lo_ecig_m  = median(real_per_mL_price_paid[total_fda_flavored_mL > 0   & total_fda_flavored_mL <= med_fda_mL & total_packs == 0],          na.rm = TRUE),
  bundle_fda_hi_ecig_m  = median(real_per_mL_price_paid[total_fda_flavored_mL > med_fda_mL & total_packs == 0],                                         na.rm = TRUE)

), keyby = purchase_month]

# Number of observations with NA for the median monthly prices
nrow(dt_month_prices[!complete.cases(dt_month_prices)])

# Get necessary columns so price data can be linked
dt_prices <- dt[, .(household_code, purchase_month, fips_state_code, fips_county_code,
                    total_packs, total_mL, total_flavored_mL, total_original_mL,
                    real_per_pack_price_paid, real_per_mL_price_paid)]
dt_prices[, names(dt_product_choices) := dt_product_choices]

# Merge county-month median prices
cm_cols <- setdiff(
  names(dt_county_month_prices),
  c("fips_county_code", "purchase_month")
)
dt_prices[
  dt_county_month_prices,
  on = .(fips_county_code, purchase_month),
  (cm_cols) := mget(paste0("i.", cm_cols))
]

# Merge state-month median prices
sm_cols <- setdiff(
  names(dt_state_month_prices),
  c("fips_state_code", "purchase_month")
)
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

# Final price columns: one price per bin per observation
# The model uses E[i,j] = p_cig[i,j] * c_cig[j] + p_ecig[i,j] * c_ecig[j]
# where prices are both observation-specific (county-month) and alternative-specific (quantity bin)
#
# Buying in bulk has lower per-unit prices (quantity discounts)
#
# Hierarchical imputation (4 tiers):
#   1. Actual price paid if the household chose alternative j
#   2. County-month median price for bin j
#   3. State-month median price for bin j
#   4. Monthly median price for bin j
# Tier 1 uses the household's actual per-unit price for the chosen alternative only;
# unchosen alternatives get NA and fall through to tiers 2/3/4.

# Tier 1: Actual price paid if alternative j was chosen (NA otherwise)
# Cigarette alternatives
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

# E-cigarette alternatives (pooled: actual price assigned if household bought any ecig in this total_mL bin)
dt_prices[, ecig_0to5_actual   := fifelse(total_mL > 0  & total_mL <= 5,  real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_5to10_actual  := fifelse(total_mL > 5  & total_mL <= 10, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_10to15_actual := fifelse(total_mL > 10 & total_mL <= 15, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_15to20_actual := fifelse(total_mL > 15 & total_mL <= 20, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_20to30_actual := fifelse(total_mL > 20 & total_mL <= 30, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_30to50_actual := fifelse(total_mL > 30 & total_mL <= 50, real_per_mL_price_paid, NA_real_)]
dt_prices[, ecig_50plus_actual := fifelse(total_mL > 50,                  real_per_mL_price_paid, NA_real_)]

# Bundle alternatives: actual cig and ecig prices paid when household chose that bundle type
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

# Final price columns: fcoalesce(actual, county-month median, state-month median, monthly median)
# Cigarette alternatives
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

# E-cigarette alternatives (pooled orig/flav; single ecig price state)
dt_prices[, ecig_0to5_p   := fcoalesce(ecig_0to5_actual, ecig_0to5_cm, ecig_0to5_sm, ecig_0to5_m)]
dt_prices[, ecig_5to10_p  := fcoalesce(ecig_5to10_actual, ecig_5to10_cm, ecig_5to10_sm, ecig_5to10_m)]
dt_prices[, ecig_10to15_p := fcoalesce(ecig_10to15_actual, ecig_10to15_cm, ecig_10to15_sm, ecig_10to15_m)]
dt_prices[, ecig_15to20_p := fcoalesce(ecig_15to20_actual, ecig_15to20_cm, ecig_15to20_sm, ecig_15to20_m)]
dt_prices[, ecig_20to30_p := fcoalesce(ecig_20to30_actual, ecig_20to30_cm, ecig_20to30_sm, ecig_20to30_m)]
dt_prices[, ecig_30to50_p := fcoalesce(ecig_30to50_actual, ecig_30to50_cm, ecig_30to50_sm, ecig_30to50_m)]
dt_prices[, ecig_50plus_p := fcoalesce(ecig_50plus_actual, ecig_50plus_cm, ecig_50plus_sm, ecig_50plus_m)]

# Bundle alternatives: cig and ecig components
# Tier 1: actual price paid by bundle purchaser; tiers 2-4: standalone bin medians
# All lo bundles share the same cig fallback (<=20 pack standalone median)
# All hi bundles share the same cig fallback (>20 pack standalone median)
# All bundles share the same ecig fallback (all ecig-only standalone median)
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

# Final price columns (one per alternative, varies by observation due to county-month)
# Standalone cig bins (j=2:13), standalone ecig bins (j=14:34, pooled orig/flav)
# and bundle components (j=35:40, separate cig and ecig prices per bundle)
price_cols <- c(
  # Standalone cig bins
  "cig_1_p", "cig_2_p", "cig_3to4_p", "cig_5to9_p", "cig_10_p", "cig_11to19_p",
  "cig_20_p", "cig_21to29_p", "cig_30_p", "cig_31to39_p", "cig_40_p", "cig_41plus_p",
  # Standalone ecig bins (pooled orig/flav)
  "ecig_0to5_p", "ecig_5to10_p", "ecig_10to15_p", "ecig_15to20_p",
  "ecig_20to30_p", "ecig_30to50_p", "ecig_50plus_p",
  # Bundle cig components (12 bundles x cig price)
  "bundle_orig_ll_cig_p", "bundle_orig_lh_cig_p", "bundle_orig_hl_cig_p", "bundle_orig_hh_cig_p",
  "bundle_nfda_ll_cig_p", "bundle_nfda_lh_cig_p", "bundle_nfda_hl_cig_p", "bundle_nfda_hh_cig_p",
  "bundle_fda_ll_cig_p",  "bundle_fda_lh_cig_p",  "bundle_fda_hl_cig_p",  "bundle_fda_hh_cig_p",
  # Bundle ecig components (12 bundles x ecig price)
  "bundle_orig_ll_ecig_p", "bundle_orig_lh_ecig_p", "bundle_orig_hl_ecig_p", "bundle_orig_hh_ecig_p",
  "bundle_nfda_ll_ecig_p", "bundle_nfda_lh_ecig_p", "bundle_nfda_hl_ecig_p", "bundle_nfda_hh_ecig_p",
  "bundle_fda_ll_ecig_p",  "bundle_fda_lh_ecig_p",  "bundle_fda_hl_ecig_p",  "bundle_fda_hh_ecig_p"
)

# Number of NAs across price columns
dt_prices[, sum(sapply(.SD, function(x) sum(is.na(x)))), .SDcols = price_cols]

# Keep only the final price columns
dt_prices <- dt_prices[, ..price_cols]

# Write the data to a file
file_name <- "../Dynamic_Model/Data/Prices.csv"
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
# Price Ratios
# (Quantity Discounts)
#############################

# Compute price ratios: median per-unit price in each bin / overall category median
# Small-quantity bins pay more per unit, large-quantity bins pay less (quantity discounts)
# Ratios are fixed constants used in the expenditure computation (get_expenditures() in Julia)

# Overall category medians (across all purchasers regardless of quantity bin)
overall_cig_price  <- dt[total_packs > 0, median(real_per_pack_price_paid, na.rm = TRUE)]
overall_ecig_price <- dt[total_mL > 0,    median(real_per_mL_price_paid, na.rm = TRUE)]

# Cigarette bin medians and ratios
dt_price_ratios <- data.table(
  alternative = c(
    # Cigarette bins (all purchasers in pack quantity range)
    "cig_1", "cig_2", "cig_3to4", "cig_5to9", "cig_10", "cig_11to19",
    "cig_20", "cig_21to29", "cig_30", "cig_31to39", "cig_40", "cig_41plus",
    # E-cig bins (pooled orig/flav, by total mL)
    "ecig_0to5", "ecig_5to10", "ecig_10to15", "ecig_15to20",
    "ecig_20to30", "ecig_30to50", "ecig_50plus"
  ),
  median_price = c(
    # Cigarette bin medians (all purchasers in pack quantity range)
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
    dt[total_mL > 50, median(real_per_mL_price_paid, na.rm = TRUE)]
  ),
  overall_median = c(
    # Cig bins use overall cig median
    rep(overall_cig_price, 12),
    # Ecig bins use overall ecig median
    rep(overall_ecig_price, 7)
  )
)

# Compute ratios
dt_price_ratios[, ratio := median_price / overall_median]
dt_price_ratios

# Write to file
file_name <- "../Dynamic_Model/Data/Price_Ratios.csv"
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
# Lagged Category Choice
# (State Dependence)
#############################

# Build household-month panel with category choice
# Raw CSV is pre-sorted by (household_code, purchase_month), so no additional sorting needed.
dt_lag <- dt[, .(household_code, purchase_month, outside_option, cig, ecig, cig_ecig)]

# Create a numeric category indicator:
#   0 = outside option, 1 = cig, 2 = ecig, 3 = cig_ecig
dt_lag[, category := fifelse(cig == 1, 1,
                    fifelse(ecig == 1, 2,
                    fifelse(cig_ecig == 1, 3, 0)))]

# Lag within household: shift category by one period
dt_lag[, lagged_category := shift(category, n = 1, type = "lag"), by = household_code]

# Create lagged category indicators for each product type
# (NA for first observation of each household; no prior choice observed)
dt_lag[, lagged_cig      := fifelse(lagged_category == 1, 1, 0)]
dt_lag[, lagged_ecig     := fifelse(lagged_category == 2, 1, 0)]
dt_lag[, lagged_cig_ecig := fifelse(lagged_category == 3, 1, 0)]

# Number of observations with no lagged choice (first month per household)
cat("Observations with no lagged choice:", dt_lag[is.na(lagged_category), .N], "\n")
cat("Observations with lagged choice:", dt_lag[!is.na(lagged_category), .N], "\n")

# Keep only the lagged indicators (order matches dt rows)
dt_lagged_choice <- dt_lag[, .(lagged_cig, lagged_ecig, lagged_cig_ecig)]

# Write the data to a file
file_name <- "../Dynamic_Model/Data/Lagged_Category_Choice.csv"
fwrite(dt_lagged_choice, file_name)

# Confirm results have been written to a file
if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Lead Prices
# (Forward-Looking Behavior)
#############################

# State-month median per-unit prices for cigarettes and e-cigarettes
dt_lead_sm_prices <- dt[, .(
  median_cig_price_sm  = median(real_per_pack_price_paid[total_packs > 0], na.rm = TRUE),
  median_ecig_price_sm = median(real_per_mL_price_paid[total_mL > 0], na.rm = TRUE)
), keyby = .(fips_state_code, purchase_month)]

# Create lead prices: next month's state-month median
dt_lead_sm_prices[, lead_cig_price_sm  := shift(median_cig_price_sm, n = 1, type = "lead"), by = fips_state_code]
dt_lead_sm_prices[, lead_ecig_price_sm := shift(median_ecig_price_sm, n = 1, type = "lead"), by = fips_state_code]

# Monthly median per-unit prices (fallback when state-month is unavailable)
dt_lead_m_prices <- dt[, .(
  median_cig_price_m  = median(real_per_pack_price_paid[total_packs > 0], na.rm = TRUE),
  median_ecig_price_m = median(real_per_mL_price_paid[total_mL > 0], na.rm = TRUE)
), keyby = purchase_month]

# Create lead prices: next month's monthly median
dt_lead_m_prices[, lead_cig_price_m  := shift(median_cig_price_m, n = 1, type = "lead")]
dt_lead_m_prices[, lead_ecig_price_m := shift(median_ecig_price_m, n = 1, type = "lead")]

# Build lead price data table with household identifiers
dt_lead <- dt[, .(household_code, purchase_month, fips_state_code)]

# Merge state-month lead prices
dt_lead[
  dt_lead_sm_prices,
  on = .(fips_state_code, purchase_month),
  `:=`(lead_cig_price_sm  = i.lead_cig_price_sm,
       lead_ecig_price_sm = i.lead_ecig_price_sm)
]

# Merge monthly lead prices
dt_lead[
  dt_lead_m_prices,
  on = .(purchase_month),
  `:=`(lead_cig_price_m  = i.lead_cig_price_m,
       lead_ecig_price_m = i.lead_ecig_price_m)
]

# Hierarchical imputation: state-month first, then monthly fallback
dt_lead[, lead_cig_price  := fcoalesce(lead_cig_price_sm, lead_cig_price_m)]
dt_lead[, lead_ecig_price := fcoalesce(lead_ecig_price_sm, lead_ecig_price_m)]

# Number of observations with no lead price (last month in sample)
cat("Observations with no lead price:", dt_lead[is.na(lead_cig_price), .N], "\n")
cat("Observations with lead price:", dt_lead[!is.na(lead_cig_price), .N], "\n")

# Only the final sample month should appear since that's the only month
# without an available lead
dt_lead[is.na(lead_cig_price), unique(purchase_month)]

# Keep only the final lead price columns
dt_lead_prices <- dt_lead[, .(lead_cig_price, lead_ecig_price)]

# Write the data to a file
file_name <- "../Dynamic_Model/Data/Lead_Prices.csv"
fwrite(dt_lead_prices, file_name)

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

# Load pricing grid (5 quintile points) to map households to price states
dt_pricing_spaces <- fread("../Dynamic_Model/Data/Pricing_Spaces.csv")
cig_grid  <- dt_pricing_spaces[, cig]
ecig_grid <- dt_pricing_spaces[, ecig]

# State-month median prices per household-month (used in model)
dt_hh_prices <- dt[, .(
  household_code,
  purchase_month,
  fips_state_code
)]

# Merge state-month median prices (same prices the model uses)
dt_sm <- dt[, .(
  median_cig_price_sm  = median(real_per_pack_price_paid[total_packs > 0], na.rm = TRUE),
  median_ecig_price_sm = median(real_per_mL_price_paid[total_mL > 0],      na.rm = TRUE)
), keyby = .(fips_state_code, purchase_month)]

dt_hh_prices[
  dt_sm,
  on = .(fips_state_code, purchase_month),
  `:=`(median_cig_price_sm  = i.median_cig_price_sm,
       median_ecig_price_sm = i.median_ecig_price_sm)
]

# Map state-month median prices to 5-point price grid states (1 = lowest, 5 = highest)
dt_hh_prices[, cig_price_state  := pmin(pmax(findInterval(median_cig_price_sm,  cig_grid),  1), 5)]
dt_hh_prices[, ecig_price_state := pmin(pmax(findInterval(median_ecig_price_sm, ecig_grid), 1), 5)]

# Within-household price variation: SD and price state transitions
dt_cig_var <- dt_hh_prices[, .(
  mean_cig_price     = mean(median_cig_price_sm,  na.rm = TRUE),
  sd_cig_price       = sd(median_cig_price_sm,    na.rm = TRUE),
  n_months           = .N,
  n_cig_states       = uniqueN(cig_price_state),
  n_state_changes    = sum(diff(cig_price_state) != 0, na.rm = TRUE)
), keyby = household_code]

dt_ecig_var <- dt_hh_prices[, .(
  mean_ecig_price    = mean(median_ecig_price_sm, na.rm = TRUE),
  sd_ecig_price      = sd(median_ecig_price_sm,   na.rm = TRUE),
  n_months           = .N,
  n_ecig_states      = uniqueN(ecig_price_state),
  n_state_changes    = sum(diff(ecig_price_state) != 0, na.rm = TRUE)
), keyby = household_code]

# Summary statistics
cat("\n--- Cigarette Price Variation (within household, state-month median) ---\n")
print(dt_cig_var[, .(
  mean_sd            = mean(sd_cig_price,     na.rm = TRUE),
  median_sd          = median(sd_cig_price,   na.rm = TRUE),
  pct_0_transitions  = mean(n_state_changes == 0) * 100,
  mean_transitions   = mean(n_state_changes),
  median_transitions = median(n_state_changes),
  mean_n_states      = mean(n_cig_states)
)])

cat("\n--- E-Cigarette Price Variation (within household, state-month median) ---\n")
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

















