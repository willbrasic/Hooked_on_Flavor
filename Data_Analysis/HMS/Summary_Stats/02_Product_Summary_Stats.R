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



