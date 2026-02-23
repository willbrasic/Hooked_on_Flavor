################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script aggregates the cleaned tobacco panelists purchases data 
# to the monthly level and
# links it to the household information data.
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
pacman::p_load(data.table, fredr, zoo)

# Set fred api key
fredr_set_key("f03899f7975d19cf161de9e27746cd26")

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 20)


#############################
# Aggregate data to 
# monthly level
#############################

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_CLEANED_2021-Onward.rds")
dt <- readRDS(file_name)

# Drop unnecessary columns
dt <- dt[, `:=` (quantity = NULL,
                 nicotine_mg_consumed_per_mL = NULL
                 )]

# Aggregate data to the monthly household level
start_col <- match("total_packs", names(dt))
end_col <- match("ecig_nicotine_mg_absorbed", names(dt))
cols_sum <- names(dt)[start_col:end_col]
dt_monthly <- dt[, c(lapply(.SD, function(x) sum(x, na.rm = TRUE)),
                    .(number_of_purchases = .N,
                      number_of_unique_categories = uniqueN(category_cd),
                      number_of_unique_segments = uniqueN(segment_cd),
                      total_original_mL = sum(total_mL[ecig == 1 & original_ecig == 1], na.rm = TRUE),
                      total_flavored_mL = sum(total_mL[ecig == 1 & flavored_ecig == 1], na.rm = TRUE),
                      total_fda_authorized_mL = sum(total_mL[ecig == 1 & fda_authorized_ecig == 1], na.rm = TRUE),
                      total_fda_flavored_mL = sum(total_mL[ecig == 1 & flavored_ecig == 1 & fda_authorized_ecig == 1], na.rm = TRUE),
                      total_non_fda_flavored_mL = sum(total_mL[ecig == 1 & flavored_ecig == 1 & fda_authorized_ecig == 0], na.rm = TRUE),
                      total_price_paid = sum(total_price_paid, na.rm = TRUE),
                      total_pack_price_paid = sum(total_price_paid[cig == 1], na.rm = TRUE),
                      total_mL_price_paid = sum(total_price_paid[ecig == 1], na.rm = TRUE))),
                by = .(household_code, purchase_month),
                .SDcols = cols_sum
][order(household_code, purchase_month)]
setcolorder(dt_monthly, c("number_of_purchases", "number_of_unique_categories", "number_of_unique_segments"),
            after = "purchase_month")
setcolorder(dt_monthly, c("consumption_in_cigs", "consumption_in_packs"),
            after = "number_of_unique_segments")
setcolorder(dt_monthly, c("total_original_mL", "total_flavored_mL", "total_fda_authorized_mL",
                          "total_fda_flavored_mL", "total_non_fda_flavored_mL"),
            after = "total_mL")

# Create total nicotine consumed and absorbed columns
dt_monthly[, `:=` (total_nicotine_mg_consumed = cig_nicotine_mg_consumed + ecig_nicotine_mg_consumed,
                   total_nicotine_mg_absorbed = cig_nicotine_mg_absorbed + ecig_nicotine_mg_absorbed
                   )]
setcolorder(dt_monthly, c("total_nicotine_mg_consumed", "total_nicotine_mg_absorbed"),
            after = "ecig_nicotine_mg_absorbed")

# Create year of purchase column
dt_monthly[, purchase_year := as.integer(format(purchase_month, "%Y"))]
setcolorder(dt_monthly, "purchase_year", after = "household_code")

# Indicator for purchase of cigs and e-cigs
dt_monthly[, `:=` (cig = fifelse(total_packs > 0, 1, 0),
                   ecig = fifelse(total_mL > 0, 1, 0)
                   )]
setcolorder(dt_monthly, c("cig", "ecig"), after = "number_of_unique_segments")

# Indicator for of household flavor purchases 
dt_monthly[, `:=` (original_ecig = fifelse(total_original_mL > 0, 1, 0),
                   flavored_ecig = fifelse(total_flavored_mL > 0, 1, 0),
                   fda_authorized_ecig = fifelse(total_fda_authorized_mL > 0, 1, 0),
                   fda_flavored_ecig = fifelse(total_fda_flavored_mL > 0, 1, 0),
                   non_fda_flavored_ecig = fifelse(total_non_fda_flavored_mL > 0, 1, 0)
                   )]
setcolorder(dt_monthly, c("original_ecig", "flavored_ecig", "fda_authorized_ecig",
                          "fda_flavored_ecig", "non_fda_flavored_ecig"), after = "ecig")

# Household-year combinations that actually exist for a given household
hh_years <- unique(dt_monthly[, .(household_code, purchase_year)])

# Build a month grid only for those household-year pairs
grid <- hh_years[, .(purchase_month = seq(as.IDate(paste0(purchase_year, "-01-01")), 
                                          as.IDate(paste0(purchase_year, "-12-01")), 
                                          by = "1 month")), 
                 by = .(household_code, purchase_year)
                 ]

# Left join monthly aggregated data onto this grid so now
# each household has the months even when a purchase was not made
setkey(dt_monthly, household_code, purchase_month)
dt_monthly <- dt_monthly[grid, on = .(household_code, purchase_month)][, purchase_year := as.integer(format(purchase_month, "%Y"))]
dt_monthly <- dt_monthly[, i.purchase_year := NULL]

# Fill zeros except for price totals, which should remain NA when no purchase
keep_na <- c("total_price_paid", "total_pack_price_paid", "total_mL_price_paid")
setcolorder(dt_monthly, c(setdiff(names(dt_monthly), keep_na), keep_na))
fill0_cols <- setdiff(names(dt_monthly), c("household_code", "purchase_month", "purchase_year", keep_na))
dt_monthly[, (fill0_cols) := lapply(.SD, function(x) 
{
  if (is.integer(x)) replace(x, is.na(x), 0L)
  else if (is.numeric(x)) replace(x, is.na(x), 0)
  else if (is.factor(x)) replace(x, is.na(x), 0)
  else x
}), .SDcols = fill0_cols]

# Enter NA for total prices when no purchase of the category was made in the month
dt_monthly[, `:=` (total_pack_price_paid = fifelse(cig == 1, total_pack_price_paid, NA_real_),
                   total_mL_price_paid = fifelse(ecig == 1, total_mL_price_paid, NA_real_)
                   )]

# Derive per-unit prices using the aggregated totals
dt_monthly[, `:=` (per_pack_price_paid = fifelse(cig == 1, total_pack_price_paid / total_packs, NA_real_),
                   per_mL_price_paid = fifelse(ecig == 1, total_mL_price_paid / total_mL, NA_real_)
                   )]

# Create outside option alternative
dt_monthly[, outside_option := fifelse(
  total_packs == 0 & total_mL == 0, 1, 0
)]
setcolorder(dt_monthly, "outside_option", after = "number_of_unique_segments")


#############################
# Examine multi-category
# purchase months
#############################

# Share of e-cigarette purchase months that contain both e-cig and cig purchases
dt_monthly[ecig == 1, .(
  share_multi_cat = mean(cig == 1 & ecig == 1, na.rm = TRUE)
)]

# Share of e-cigarette purchase months that contain both original and flavored e-cig purchases
dt_monthly[ecig == 1, .(
  share_multi_cat = mean(flavored_ecig == 1 & original_ecig == 1, na.rm = TRUE)
)]

# Make flavored and original e-cig columns mutually exclusive based on whether a 
# household consumed more flavored or more original in that month
dt_monthly[, flavored_ecig := as.integer(as.character(flavored_ecig))]
dt_monthly[, original_ecig := as.integer(as.character(original_ecig))]
dt_monthly[
  flavored_ecig == 1 & original_ecig == 1,
  c(
    "flavored_ecig",
    "original_ecig",
    "total_flavored_mL",
    "total_original_mL"
  ) := list(
    as.integer(total_flavored_mL >= total_original_mL),
    as.integer(total_flavored_mL <  total_original_mL),
    fifelse(total_flavored_mL >= total_original_mL,
            total_mL, 0),
    fifelse(total_flavored_mL < total_original_mL,
            total_mL, 0)
  )
]

# Zero out FDA/non-FDA flavored mL when original won the resolution
dt_monthly[flavored_ecig == 0, `:=`(
  total_fda_flavored_mL = 0,
  total_non_fda_flavored_mL = 0,
  fda_flavored_ecig = 0,
  non_fda_flavored_ecig = 0
)]

# Resolve FDA vs non-FDA within flavored months
dt_monthly[
  flavored_ecig == 1 & fda_flavored_ecig == 1 & non_fda_flavored_ecig == 1,
  c("fda_flavored_ecig", "non_fda_flavored_ecig",
    "total_fda_flavored_mL", "total_non_fda_flavored_mL") := list(
    as.integer(total_fda_flavored_mL >= total_non_fda_flavored_mL),
    as.integer(total_fda_flavored_mL <  total_non_fda_flavored_mL),
    fifelse(total_fda_flavored_mL >= total_non_fda_flavored_mL, total_flavored_mL, 0),
    fifelse(total_fda_flavored_mL <  total_non_fda_flavored_mL, total_flavored_mL, 0)
  )
]

# When only one type is present, assign all flavored mL to that type
dt_monthly[flavored_ecig == 1 & fda_flavored_ecig == 1 & non_fda_flavored_ecig == 0,
           total_fda_flavored_mL := total_flavored_mL]
dt_monthly[flavored_ecig == 1 & fda_flavored_ecig == 0 & non_fda_flavored_ecig == 1,
           total_non_fda_flavored_mL := total_flavored_mL]

# # Mutual exclusivity checks (each one should print zero)
# dt_monthly[fda_flavored_ecig == 1 & non_fda_flavored_ecig == 1, .N]
# dt_monthly[flavored_ecig == 0 & (fda_flavored_ecig == 1 | non_fda_flavored_ecig == 1), .N]
# dt_monthly[flavored_ecig == 1 & (fda_flavored_ecig + non_fda_flavored_ecig) != 1, .N]
# dt_monthly[fda_flavored_ecig == 0 & total_fda_flavored_mL > 0, .N]
# dt_monthly[non_fda_flavored_ecig == 0 & total_non_fda_flavored_mL > 0, .N]
# dt_monthly[fda_flavored_ecig == 1 & total_fda_flavored_mL == 0, .N]
# dt_monthly[non_fda_flavored_ecig == 1 & total_non_fda_flavored_mL == 0, .N]
# dt_monthly[flavored_ecig == 1 & total_fda_flavored_mL + total_non_fda_flavored_mL !=
#              total_flavored_mL, .N]

# Define (cig, e-cig) category
dt_monthly[, `:=`(
  cig_ecig = fifelse(cig == 1 & ecig == 1, 1, 0),
  cig_tmp  = cig,
  ecig_tmp = ecig
)]
dt_monthly[, `:=`(
  cig  = fifelse(cig_tmp == 1 & ecig_tmp == 0, 1, 0),
  ecig = fifelse(cig_tmp == 0 & ecig_tmp == 1, 1, 0)
)]
dt_monthly[, c("cig_tmp", "ecig_tmp") := NULL]
setcolorder(dt_monthly, "cig_ecig", after = "ecig")

# # Ensure categories are mutually exclusive
# dt_monthly[(cig + ecig + cig_ecig) > 1, .N]
# 
# # Ensure categories are free of NAs
# dt_monthly[
#   rowSums(
#     cbind(cig, ecig, cig_ecig),
#     na.rm = TRUE
#   ) > 1,
#   .N
# ]

#############################
# Tobacco product cpi data
# to get real prices
#############################

# Get tobacco cpi data from fred
dt_cpi <- data.table(fredr(
  series_id = "CUSR0000SEGA",
  observation_start = min(dt_monthly[, purchase_month]),
  observation_end = max(dt_monthly[, purchase_month]),
  frequency = "m"
))
setDT(dt_cpi)
dt_cpi[, purchase_month := as.IDate(as.Date(as.yearmon(date)))]
dt_cpi <- dt_cpi[, .(purchase_month, cpi = value)]

# Merge cpi data to monthly panel data
dt_monthly <- merge(dt_monthly, dt_cpi, by = "purchase_month", all.x = TRUE)[order(household_code, purchase_month)]

# Compute real per-unit prices
base_cpi <- dt_cpi[1, cpi]
dt_monthly[cig == 1 | cig_ecig == 1, real_per_pack_price_paid := per_pack_price_paid * (base_cpi / cpi)]
dt_monthly[ecig == 1 | cig_ecig == 1, real_per_mL_price_paid := per_mL_price_paid * (base_cpi / cpi)]
dt_monthly[, cpi := NULL]

# Ensure there aren't any NAs for the real per-unit prices
nrow(dt_monthly[(cig == 1 | cig_ecig == 1) & is.na(real_per_pack_price_paid)])
nrow(dt_monthly[(ecig == 1 | cig_ecig == 1) & is.na(real_per_mL_price_paid)])


#############################
# Link panelist information
# to the aggregated monthly
# panel
#############################

# Load in household information data
file_name <- paste0("../tobacco_panelists_information_2021-Onward.rds")
dt_households <- readRDS(file_name)

# Build HH-year grid from union of dt_monthly and dt_households
hh_years <- unique(rbind(
  dt_monthly[, .(household_code, purchase_year)],
  dt_households[, .(household_code, purchase_year)]
))

# Join dt_households onto that grid
hh_panel <- merge(
  hh_years,
  dt_households,
  by = c("household_code", "purchase_year"),
  all.x = TRUE
)

# Fill missing years within household (forward then backward)
setorder(hh_panel, household_code, purchase_year)

hh_cols <- setdiff(names(dt_households), c("household_code", "purchase_year"))

hh_panel[, (hh_cols) := lapply(.SD, zoo::na.locf, na.rm = FALSE),
         by = household_code, .SDcols = hh_cols]

hh_panel[, (hh_cols) := lapply(.SD, function(x) zoo::na.locf(x, fromLast = TRUE, na.rm = FALSE)),
         by = household_code, .SDcols = hh_cols]

# Merge filled HH-year panel into dt_monthly
dt_monthly <- merge(
  dt_monthly,
  hh_panel,
  by = c("household_code", "purchase_year"),
  all.x = TRUE
)


#############################
# Clean up monthly 
# aggregated data
#############################

# Number of households where teen or young adult presence changes over time
dt_monthly[
  , .(n_unique = uniqueN(teen_or_young_adult_present)),
  by = household_code
][n_unique > 1, .N]

# Teen or young adult indicator that is constant within households
dt_monthly[, teen_or_young_adult_ever := as.integer(
  any(teen_or_young_adult_present == 1, na.rm = TRUE)
), by = household_code]

# Teen or young adult indicator for when a household has one for all years
dt_monthly[, teen_or_young_adult_always := as.integer(
  all(teen_or_young_adult_present == 1, na.rm = TRUE)
), by = household_code]

# Rearrange column order
setcolorder(dt_monthly, c("teen_or_young_adult_ever", "teen_or_young_adult_always"), 
            after = "teen_or_young_adult_present")

# Make sure no household has NAs for any of their characteristics
# besides the member age columns, male had age, and female head age
dt_monthly[, lapply(.SD, function(x) sum(is.na(x)))]


#############################
# Output aggregated monthly
# data to a file
#############################

# Write the full panel data table to a file
file_name_csv <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
file_name_rds <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.rds")
fwrite(dt_monthly, file_name_csv)
saveRDS(dt_monthly, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds)) 
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else 
{
  cat("Error: File could not be written\n")
}












