################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script aggregates the cleaned tobacco panelists purchases data
# at "./tobacco_panelists_purchases_CLEANED_2004-2020.csv" 
# to the monthly level and links it to the household information data at 
# "../tobacco_panelists_information_2004-2020.csv."
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
                "4th_Year_Paper_Data/HMS/2004-2020/Tobacco_Panelists_Purchases_2004-2020/")
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
options(datatable.print.nrows = 999)


#############################
# Aggregate data to 
# monthly level
#############################

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_CLEANED_2004-2020.rds")
dt <- readRDS(file_name)

# Drop unnecessary columns
dt <- dt[, `:=` (quantity = NULL,
                 nicotine_mg_per_mL = NULL,
                 nicotine_mg_per_cessation = NULL
)]

# Aggregate data to the monthly household level
start_col <- match("total_packs", names(dt))
# end_col <- match("nicotine_mg_absorbed_in_cessation", names(dt))
end_col <- match("nicotine_mg_absorbed_in_packs", names(dt))
cols_sum <- names(dt)[start_col:end_col]
dt_monthly <- dt[, c(lapply(.SD, function(x) sum(x, na.rm = TRUE)),
                     .(number_of_purchases = .N,
                       total_original_packs = sum(total_packs[cig == 1], na.rm = TRUE),
                       # total_menthol_packs = sum(total_packs[cig == 1 & (menthol == 1 | flavored == 1)], na.rm = TRUE),
                       # total_original_mL = sum(total_mL[e_cig == 1 & original == 1], na.rm = TRUE),
                       # total_flavored_mL = sum(total_mL[e_cig == 1 & (menthol == 1 | flavored == 1)], na.rm = TRUE),
                       # total_original_cessation = sum(total_cessation[cessation == 1 & original == 1], na.rm = TRUE),
                       # total_flavored_cessation = sum(total_cessation[cessation == 1 & (menthol == 1 | flavored == 1)], na.rm = TRUE),
                       total_price_paid = sum(total_price_paid, na.rm = TRUE),
                       total_pack_price_paid = sum(total_price_paid[cig == 1], na.rm = TRUE)
                       # total_mL_price_paid = sum(total_price_paid[e_cig == 1], na.rm = TRUE),
                       # total_cessation_price_paid = sum(total_price_paid[cessation == 1], na.rm = TRUE)
                     )),
                 by = .(household_code, purchase_month),
                 .SDcols = cols_sum
][order(household_code, purchase_month)]
# setcolorder(dt_monthly, c("number_of_purchases", "number_of_unique_categories", "number_of_unique_segments"),
#             after = "purchase_month")
# setcolorder(dt_monthly, c("consumption_in_cigs", "consumption_in_packs"),
#             after = "number_of_unique_segments")
# setcolorder(dt_monthly, c("total_original_packs", "total_menthol_packs"),
#             after = "total_packs")
# setcolorder(dt_monthly, c("total_original_mL", "total_flavored_mL"),
#             after = "total_mL")
# setcolorder(dt_monthly, c("total_original_cessation", "total_flavored_cessation"),
#             after = "total_cessation")

# Create year of purchase column
dt_monthly[, purchase_year := as.integer(format(purchase_month, "%Y"))]
setcolorder(dt_monthly, "purchase_year", after = "household_code")

# Drop all observations that aren't in the years we are interested in
dt_monthly <- dt_monthly[purchase_year %in% c(2018, 2019, 2020)]

# Indicator for purchase of cigs, e-cigs, and cessation 
dt_monthly[, `:=` (cig = fifelse(total_packs > 0, 1, 0)
                   # e_cig = fifelse(total_mL > 0, 1, 0),
                   # cessation = fifelse(total_cessation > 0, 1, 0)
)]
# cols_factor <- c("cig", "e_cig", "cessation")
cols_factor <- c("cig")
dt_monthly[, (cols_factor) := lapply(.SD, as.factor), .SDcols = cols_factor]
# setcolorder(dt_monthly, cols_factor, after = "number_of_unique_segments")

# # Indicator for of household flavor purchases purchase
# dt_monthly[, `:=` (original_pack = fifelse(total_original_packs > 0, 1, 0),
#                    menthol_pack = fifelse(total_menthol_packs > 0, 1, 0),
#                    original_e_cig = fifelse(total_original_mL > 0, 1, 0),
#                    flavored_e_cig = fifelse(total_flavored_mL > 0, 1, 0),
#                    original_cessation = fifelse(total_original_cessation > 0, 1, 0),
#                    flavored_cessation = fifelse(total_flavored_cessation > 0, 1, 0)
# )]
# cols_factor <- c(
#   "original_pack", "menthol_pack", 
#   "original_e_cig", "flavored_e_cig",
#   "original_cessation","flavored_cessation"
# )
# dt_monthly[, (cols_factor) := lapply(.SD, as.factor), .SDcols = cols_factor]
# setcolorder(dt_monthly, cols_factor, after = "cessation")

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
keep_na <- c("total_price_paid", "total_pack_price_paid"
             # "total_mL_price_paid", "total_cessation_price_paid"
)
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
dt_monthly[, `:=` (total_pack_price_paid = fifelse(cig == 1, total_pack_price_paid, NA_real_)
                   # total_mL_price_paid = fifelse(e_cig == 1, total_mL_price_paid, NA_real_),
                   # total_cessation_price_paid = fifelse(cessation == 1, total_cessation_price_paid, NA_real_)
)]

# Derive per-unit prices using the aggregated totals
dt_monthly[, `:=` (per_pack_price_paid = fifelse(cig == 1, total_pack_price_paid / total_packs, NA_real_)
                   # per_mL_price_paid = fifelse(e_cig == 1, total_mL_price_paid / total_mL, NA_real_),
                   # per_cessation_price_paid = fifelse(cessation == 1, total_cessation_price_paid / total_cessation, NA_real_)
)]

# Create outside option alternative
dt_monthly[, outside_option := factor(fifelse(total_packs == 0, 1, 0), levels = c(1, 0))]
# setcolorder(dt_monthly, "outside_option", after = "number_of_unique_segments")


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
dt_monthly[cig == 1, real_per_pack_price_paid := per_pack_price_paid * (base_cpi / cpi)]
# dt_monthly[e_cig == 1, real_per_mL_price_paid := per_mL_price_paid * (base_cpi / cpi)]
# dt_monthly[cessation == 1, real_per_cessation_price_paid := per_cessation_price_paid * (base_cpi / cpi)]
dt_monthly[, cpi := NULL]


#############################
# Link panelist information
# to the aggregated monthly
# panel
#############################

# Load in household information data
file_name <- paste0("../tobacco_panelists_information_2004-2020.rds")
dt_households <- readRDS(file_name)

# Merge household information into monthly panel data
dt_monthly <- merge(
  dt_monthly,
  dt_households,
  by = c("household_code", "purchase_year"),
  all.x = TRUE
)

# Mean income over years household is in panel
dt_monthly[, mean_household_income := mean(household_income, na.rm = TRUE), by = household_code]
setcolorder(dt_monthly, "mean_household_income", after = "household_income")

# Make teen or young adult constant across years within households
dt_monthly[, teen_or_young_adult_present := as.integer(any(teen_or_young_adult_present == 1)), 
           by = household_code]


#############################
# Output aggregated monthly
# data to a file
#############################

# Write the full panel data table to a file
file_name_csv <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2004-2020.csv")
file_name_rds <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2004-2020.rds")
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























