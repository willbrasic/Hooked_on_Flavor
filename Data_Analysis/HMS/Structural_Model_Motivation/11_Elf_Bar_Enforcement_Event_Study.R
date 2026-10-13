################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# September 2026
#
# Event study around the FDA's 2023 enforcement actions against Elf Bar
# (import alerts and manufacturer warning letters in May 2023, retailer warning
# letters on June 22, 2023).
# Compares households that bought a brand named in those actions (Elf Bar, Esco
# Bars, Breeze) between January 2022 and April 2023 with households that bought
# other unauthorized flavored e-cigs over the same window,
# tracking abstention (no tobacco purchase), cigarette purchases, and own-brand,
# authorized, and unauthorized e-cig purchases month by month in 2023.
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
pacman::p_load(data.table, fixest)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Image output directory
output_directory <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/",
                              "4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures/")

# Load in purchase-level data (for brands)
file_name <- paste0("./tobacco_panelists_purchases_CLEANED_2021-Onward.csv")
dt_purchases <- fread(file_name)

# Load in household-month panel data (for outcomes)
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt <- fread(file_name)

# Household with bulk e-liquid bottle purchases that dominate monthly totals
bulk_hh <- 30262994


#############################
# Exposed and comparison
# households
#############################

# E-cig purchases with brand
dt_purchases <- dt_purchases[!is.na(total_mL) & total_mL > 0 & household_code != bulk_hh]
dt_purchases[, purchase_month := as.IDate(purchase_month)]
dt_purchases[, brand := toupper(BRAND)]
dt_purchases[, elf_bar := grepl("ELF BAR", brand)]

# Brands named in the FDA's May 2023 actions: Elf Bar (import alert), Esco Bars
# (import alert and warning letter), and Breeze (warning letter)
dt_purchases[, targeted := grepl("ELF BAR|ESCO|BREEZE", brand)]
dt_purchases[, unauthorized := fda_authorized_ecig == 0]
dt_purchases[, brand_family := fifelse(elf_bar, "ELF BAR", fifelse(grepl("ESCO", brand), "ESCO BARS", fifelse(grepl("BREEZE", brand), "BREEZE", brand)))]

# Selection window: January 2022 to April 2023, before the May 2023 actions
dt_selection <- dt_purchases[purchase_month >= as.IDate("2022-01-01") &
                               purchase_month <= as.IDate("2023-04-01")]

# Exposed: bought a targeted brand in the selection window
hh_elf <- dt_selection[targeted == TRUE, unique(household_code)]

# Comparison: bought an unauthorized flavored brand that was not targeted in the selection window
hh_other <- setdiff(dt_selection[unauthorized == TRUE & flavored_ecig == 1 & targeted == FALSE, unique(household_code)], hh_elf)

# Own brand: modal targeted brand for exposed households; modal unauthorized brand for comparison households
dt_own_brand <- rbind(
  dt_selection[household_code %in% hh_elf & targeted == TRUE, .N,
               by = .(household_code, brand_family)][order(household_code, -N)][
                 , .SD[1], by = household_code][, .(household_code, own_brand = brand_family)],
  dt_selection[household_code %in% hh_other & unauthorized == TRUE, .N,
               by = .(household_code, brand_family)][order(household_code, -N)][
                 , .SD[1], by = household_code][, .(household_code, own_brand = brand_family)]
)

cat("Exposed (targeted-brand) households:", length(hh_elf),
    " Comparison (other unauthorized) households:", length(hh_other), "\n")


#############################
# Household-month outcomes
# for 2023
#############################

# Restrict to 2023 household-months for the two groups
dt[, purchase_month := as.IDate(purchase_month)]
dt_es <- dt[purchase_year == 2023 & household_code %in% dt_own_brand$household_code]
dt_es <- merge(dt_es, dt_own_brand, by = "household_code")

# Outcomes (unauthorized means non-FDA-authorized flavored e-cig)
dt_es[, outside_option := as.integer(cig == 0 & ecig == 0 & cig_ecig == 0)]
dt_es[, any_cig := as.integer(cig == 1 | cig_ecig == 1)]
dt_es[, any_ecig := as.integer(ecig == 1 | cig_ecig == 1)]
dt_es[, any_authorized := as.integer(total_fda_authorized_mL > 0)]
dt_es[, any_unauthorized := as.integer(total_non_fda_flavored_mL > 0)]

# Own-brand purchase in the month
dt_own_months <- dt_purchases[year(purchase_month) == 2023, .(household_code, purchase_month, brand_family)]
dt_own_months <- merge(dt_own_months, dt_own_brand, by = "household_code")
dt_own_months <- unique(dt_own_months[brand_family == own_brand, .(household_code, purchase_month, any_own_brand = 1L)])
dt_es <- merge(dt_es, dt_own_months, by = c("household_code", "purchase_month"), all.x = TRUE)
dt_es[is.na(any_own_brand), any_own_brand := 0L]

# Treatment, post period (June 2023 onward), and month index
dt_es[, treated := as.integer(household_code %in% hh_elf)]
dt_es[, post := as.integer(purchase_month >= as.IDate("2023-06-01"))]
dt_es[, month_index := month(purchase_month)]


#############################
# Difference-in-differences
#############################

# Post x treated with household and month fixed effects, clustered by household
did_outside      <- feols(outside_option ~ post:treated | household_code + month_index,
                          data = dt_es, cluster = ~household_code)
did_any_cig      <- feols(any_cig ~ post:treated | household_code + month_index,
                          data = dt_es, cluster = ~household_code)
did_any_ecig     <- feols(any_ecig ~ post:treated | household_code + month_index,
                          data = dt_es, cluster = ~household_code)
did_authorized   <- feols(any_authorized ~ post:treated | household_code + month_index,
                          data = dt_es, cluster = ~household_code)
did_unauthorized <- feols(any_unauthorized ~ post:treated | household_code + month_index,
                          data = dt_es, cluster = ~household_code)
did_own_brand    <- feols(any_own_brand ~ post:treated | household_code + month_index,
                          data = dt_es, cluster = ~household_code)

etable(did_outside, did_any_cig, did_any_ecig, did_authorized, did_unauthorized, did_own_brand,
       headers = c("Outside option", "Any cig", "Any e-cig", "Any authorized", "Any unauthorized flavored", "Own brand"), digits = 3)


#############################
# Event study
#############################

# Month x treated interactions with May 2023 as the reference month
es_outside      <- feols(outside_option ~ i(month_index, treated, ref = 5) | household_code + month_index,
                         data = dt_es, cluster = ~household_code)
es_own_brand    <- feols(any_own_brand ~ i(month_index, treated, ref = 5) | household_code + month_index,
                         data = dt_es, cluster = ~household_code)
es_unauthorized <- feols(any_unauthorized ~ i(month_index, treated, ref = 5) | household_code + month_index,
                         data = dt_es, cluster = ~household_code)

# Pull coefficients and 95% confidence intervals into a data table
get_es_coefs <- function(model, label)
{
  dt_coefs <- as.data.table(coeftable(model), keep.rownames = "term")
  setnames(dt_coefs, c("term", "estimate", "se", "t_stat", "p_value"))
  dt_coefs[, month_index := as.integer(gsub("month_index::|:treated", "", term))]
  dt_coefs <- rbind(dt_coefs[, .(month_index, estimate, se)],
                    data.table(month_index = 5L, estimate = 0, se = 0))
  dt_coefs[, `:=` (lo = estimate - 1.96 * se, hi = estimate + 1.96 * se)]
  setnames(dt_coefs, c("estimate", "lo", "hi"), paste0(label, c("_est", "_lo", "_hi")))
  dt_coefs[, se := NULL]
  dt_coefs
}

dt_es_coefs <- merge(get_es_coefs(es_outside, "outside"),
                     get_es_coefs(es_own_brand, "own_brand"), by = "month_index")
dt_es_coefs <- merge(dt_es_coefs, get_es_coefs(es_unauthorized, "unauthorized"), by = "month_index")
dt_es_coefs[, purchase_month := as.IDate(sprintf("2023-%02d-01", month_index))]
setcolorder(dt_es_coefs, "purchase_month")
setorder(dt_es_coefs, month_index)
dt_es_coefs


#############################
# Write results to a file
#############################

# Write results to a file
file_name <- paste0(output_directory, "/11_Elf_Bar_Event_Study.txt")
fwrite(
  dt_es_coefs,
  file = file_name,
  quote = FALSE
)

# Confirm results have been written
if (file.exists(file_name))
{
  cat("Results have been written to", file_name, "\n")
} else
{
  cat("Error: Figures could not be written\n")
}


#############################
# Panel-wide shares in 2023
#############################

# Household-months with a flavored e-cig purchase in 2023, matching the
# construction of the FDA authorization frequency figure
dt_flav_months <- dt[purchase_year == 2023 & (ecig == 1 | cig_ecig == 1) & flavored_ecig == 1]

# Household-months with an Elf Bar purchase (Elf Bar proper, excluding the
# manufacturer's Lost Mary, Funky Republic, and EB Create lines, some of which
# carry the Elf Bar brand code)
dt_purchases[, product := toupper(product_descr)]
dt_purchases[, elf_bar_proper := elf_bar & !grepl("LOST MARY|FUNKY|CREATE", product)]
dt_elf_months <- unique(dt_purchases[elf_bar_proper == TRUE, .(household_code, purchase_month, elf_bar_month = 1L)])
dt_flav_months <- merge(dt_flav_months, dt_elf_months, by = c("household_code", "purchase_month"), all.x = TRUE)
dt_flav_months[is.na(elf_bar_month), elf_bar_month := 0L]

# Household-months with a Lost Mary or Funky Republic purchase, the same
# manufacturer's lines that were not named in the May and June 2023 actions
dt_purchases[, lost_mary_funky := grepl("LOST MARY|FUNKY|ELFLIQ", brand) | grepl("LOST MARY|FUNKY", product)]
dt_lmf_months <- unique(dt_purchases[lost_mary_funky == TRUE, .(household_code, purchase_month, lost_mary_funky_month = 1L)])
dt_flav_months <- merge(dt_flav_months, dt_lmf_months, by = c("household_code", "purchase_month"), all.x = TRUE)
dt_flav_months[is.na(lost_mary_funky_month), lost_mary_funky_month := 0L]

# Household-months with an unauthorized flavored purchase from any brand other than Elf Bar
dt_non_elf_months <- unique(dt_purchases[unauthorized == TRUE & flavored_ecig == 1 & elf_bar_proper == FALSE, .(household_code, purchase_month, non_elf_unauth_month = 1L)])
dt_flav_months <- merge(dt_flav_months, dt_non_elf_months, by = c("household_code", "purchase_month"), all.x = TRUE)
dt_flav_months[is.na(non_elf_unauth_month), non_elf_unauth_month := 0L]

# Unauthorized share of flavored e-cig months, the share with an unauthorized brand
# other than Elf Bar, and Elf Bar and Lost Mary / Funky Republic shares of
# unauthorized flavored e-cig months, by month over 2023
dt_shares <- dt_flav_months[, .(
  flavored_months              = .N,
  unauthorized_flavored_months = sum(fda_authorized_ecig == 0),
  elf_bar_months               = sum(elf_bar_month),
  lost_mary_funky_months       = sum(lost_mary_funky_month),
  elf_bar_share_unauth         = sum(elf_bar_month) / sum(fda_authorized_ecig == 0),
  lost_mary_funky_share_unauth = sum(lost_mary_funky_month) / sum(fda_authorized_ecig == 0),
  unauthorized_share_flavored  = mean(fda_authorized_ecig == 0),
  non_elf_unauth_share_flavored = mean(non_elf_unauth_month)
), keyby = purchase_month]
dt_shares


#############################
# Binned event study
#############################

# Two-month bins around the May 2023 actions, with May alone as the reference:
# Jan-Feb, Mar-Apr, May, Jun-Jul, Aug-Sep, Oct-Nov, Dec
dt_es[, month_bin := fifelse(month_index <= 2, 1L,
                     fifelse(month_index <= 4, 3L,
                     fifelse(month_index == 5, 5L,
                     fifelse(month_index <= 7, 6L,
                     fifelse(month_index <= 9, 8L,
                     fifelse(month_index <= 11, 10L, 12L))))))]

es_outside_bins <- feols(outside_option ~ i(month_bin, treated, ref = 5) | household_code + month_index,
                         data = dt_es, cluster = ~household_code)

# Pull coefficients and 95% confidence intervals, placed at the midpoint of each bin
dt_bins <- as.data.table(coeftable(es_outside_bins), keep.rownames = "term")
setnames(dt_bins, c("term", "estimate", "se", "t_stat", "p_value"))
dt_bins[, month_bin := as.integer(gsub("month_bin::|:treated", "", term))]
dt_bins <- rbind(dt_bins[, .(month_bin, estimate, se)],
                 data.table(month_bin = 5L, estimate = 0, se = 0))
dt_bins[, `:=` (lo = estimate - 1.96 * se, hi = estimate + 1.96 * se)]
dt_bins[, bin_label := c(`1` = "Jan-Feb", `3` = "Mar-Apr", `5` = "May", `6` = "Jun-Jul",
                         `8` = "Aug-Sep", `10` = "Oct-Nov", `12` = "Dec")[as.character(month_bin)]]
dt_bins[, bin_date := as.IDate(c(`1` = "2023-01-16", `3` = "2023-03-16", `5` = "2023-05-01", `6` = "2023-06-16",
                                 `8` = "2023-08-16", `10` = "2023-10-16", `12` = "2023-12-01")[as.character(month_bin)])]
setcolorder(dt_bins, c("bin_date", "bin_label", "month_bin"))
setorder(dt_bins, month_bin)
dt_bins[, se := NULL]
dt_bins


#############################
# Write results to a file
#############################

# Write results to a file
file_name <- paste0(output_directory, "/11_Elf_Bar_Event_Study_Bins.txt")
fwrite(
  dt_bins,
  file = file_name,
  quote = FALSE
)

# Confirm results have been written
if (file.exists(file_name))
{
  cat("Results have been written to", file_name, "\n")
} else
{
  cat("Error: Figures could not be written\n")
}


#############################
# Write results to a file
#############################

# Write results to a file
file_name <- paste0(output_directory, "/11_Elf_Bar_Shares_2023.txt")
fwrite(
  dt_shares,
  file = file_name,
  quote = FALSE
)

# Confirm results have been written
if (file.exists(file_name))
{
  cat("Results have been written to", file_name, "\n")
} else
{
  cat("Error: Figures could not be written\n")
}
