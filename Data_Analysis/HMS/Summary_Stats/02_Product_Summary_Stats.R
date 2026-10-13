################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
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
pacman::p_load(data.table, ggplot2)

# CA flavor ban date and all states with flavor bans (FIPS codes)
ban_date             <- as.IDate("2022-11-01")
list_fips_ban_states <- c(6, 25, 34, 36, 44, 49)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_CLEANED_2021-Onward.csv")
dt <- fread(file_name)

# Get category specific purchase data tables
dt_cig <- dt[cig == 1]
dt_ecig <- dt[ecig == 1]

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt_monthly <- fread(file_name)

# Get category specific purchase data tables
dt_monthly_cig <- dt_monthly[cig == 1 | cig_ecig == 1]
dt_monthly_ecig <- dt_monthly[ecig == 1 | cig_ecig == 1]


#############################
# Full panel   
############################# 

# Number of unique UPCs
dt[, uniqueN(upc)]
dt_cig[, uniqueN(upc)]
dt_ecig[, uniqueN(upc)]

# Distribution of per-unit prices
summary(dt_monthly_cig[, per_pack_price_paid])
summary(dt_monthly_ecig[, per_pack_price_paid])

#############################
# Monthly panel   
############################# 

# Median monthly consumption
dt_monthly_cig[, median(total_packs, na.rm = TRUE)]
dt_monthly_ecig[, median(total_mL, na.rm = TRUE)]

# Fraction of e-cig household-purchase months with FDA authorized vs non-FDA
dt_monthly_ecig[, .(
  n_ecig_months = .N,
  n_fda_authorized = sum(fda_authorized_ecig == 1),
  n_non_fda = sum(fda_authorized_ecig == 0),
  frac_fda_authorized = mean(fda_authorized_ecig == 1),
  frac_non_fda = mean(fda_authorized_ecig == 0)
)]

# FDA authorized fraction by year
dt_monthly_ecig[, .(
  n_ecig_months = .N,
  frac_fda_authorized = mean(fda_authorized_ecig == 1),
  frac_non_fda = mean(fda_authorized_ecig == 0)
), keyby = .(flavored_ecig, purchase_year)]

# FDA authorized fraction by year, flavor, and teen/young adult presence
dt_monthly_ecig[, .(
  n_ecig_months = .N,
  frac_fda_authorized = mean(fda_authorized_ecig == 1),
  frac_non_fda = mean(fda_authorized_ecig == 0)
), keyby = .(teen_or_young_adult_present, flavored_ecig, purchase_year)]

# FDA authorized fraction by flavor type (among e-cig months)
dt_monthly_ecig[, .(
  n_ecig_months = .N,
  frac_fda_authorized = mean(fda_authorized_ecig == 1)
), keyby = flavored_ecig]

# Among FDA authorized months: median mL consumed
dt_monthly_ecig[fda_authorized_ecig == 1, median(total_mL, na.rm = TRUE)]
dt_monthly_ecig[fda_authorized_ecig == 0, median(total_mL, na.rm = TRUE)]

# Share of unique e-cig households that ever purchase FDA authorized
dt_monthly_ecig[, .(ever_fda = as.integer(any(fda_authorized_ecig == 1))),
                by = household_code][, mean(ever_fda)]


#############################
# E[q_cig_std | cig]
#############################

# q_cig_max is the top bin median (cig_41plus = 60 packs) from Consumption_Spaces.csv
q_cig_max <- 60

# E[q_cig_std | cig]: mean standardized quantity across cig household-months.
# total_packs > 60 are mapped to the top bin (capped at 1 in standardized units).
dt_monthly_cig[, mean(pmin(total_packs, q_cig_max) / q_cig_max, na.rm = TRUE)]


#############################
# Brand concentration, authorized vs. unauthorized flavored e-cigs
#############################

# Restrict to flavored e-cig purchases, the segment relevant to the flavor ban
dt_flav <- dt[ecig == 1 & flavored_ecig == 1]

# Unauthorized brand shares by number of purchases (the paper's table) and by
# total mL (for reference). Purchase counts are used because a single bulk
# buyer can dominate an mL-based share. Nielsen records the same brand under
# several distributor codes (e.g. "HQD" and "HQD (SHENZHEN HQD TECHNOLOGY)",
# "BLU ECIGS", "MY BLU", and "BLU (FONTEM)"), so the parenthesized distributor
# suffix is dropped and the Blu variants are combined before computing shares.
dt_unauth <- dt_flav[fda_authorized_ecig == 0]
dt_unauth[, BRAND_merged := trimws(sub(" \\(.*\\)$", "", BRAND))]
dt_unauth[BRAND_merged %in% c("BLU ECIGS", "MY BLU"), BRAND_merged := "BLU"]
dt_brand_shares_unauth <- dt_unauth[, .(total_mL = sum(total_mL, na.rm = TRUE), n_purchases = .N), by = BRAND_merged]
dt_brand_shares_unauth[, share_mL := total_mL / sum(total_mL)]
dt_brand_shares_unauth[, share_n := n_purchases / sum(n_purchases)]
setorder(dt_brand_shares_unauth, -share_n)
print(dt_brand_shares_unauth[1:20])

cat("Top 5 unauthorized brands cumulative mL share:", sum(dt_brand_shares_unauth$share_mL[1:5]), "\n")
cat("Top 10 unauthorized brands cumulative mL share:", sum(dt_brand_shares_unauth$share_mL[1:10]), "\n")
cat("Number of distinct unauthorized brands:", nrow(dt_brand_shares_unauth), "\n")
cat("HHI (unauthorized, purchase-based, x10000 scale):", sum((dt_brand_shares_unauth$share_n)^2) * 10000, "\n")
cat("HHI (unauthorized, mL-based, x10000 scale):", sum((dt_brand_shares_unauth$share_mL)^2) * 10000, "\n")
cat("Share of unauthorized purchases outside the top 4 brands:", 1 - sum(dt_brand_shares_unauth$share_n[1:4]), "\n")

# Authorized brand shares (by total mL), for comparison
dt_auth <- dt_flav[fda_authorized_ecig == 1]
dt_brand_shares_auth <- dt_auth[, .(total_mL = sum(total_mL, na.rm = TRUE), n_purchases = .N), by = BRAND]
dt_brand_shares_auth[, share_mL := total_mL / sum(total_mL)]
dt_brand_shares_auth[, share_n := n_purchases / sum(n_purchases)]
setorder(dt_brand_shares_auth, -share_n)
print(dt_brand_shares_auth)
cat("HHI (authorized, purchase-based, x10000 scale):", sum((dt_brand_shares_auth$share_n)^2) * 10000, "\n")
cat("HHI (authorized, mL-based, x10000 scale):", sum((dt_brand_shares_auth$share_mL)^2) * 10000, "\n")

# Unauthorized share of the overall flavored e-cig market in this sample
cat("Unauthorized share of flavored mL:", sum(dt_unauth$total_mL, na.rm = TRUE) / sum(dt_flav$total_mL, na.rm = TRUE), "\n")



