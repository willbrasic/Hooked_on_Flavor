################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script combines the 2004-2020 HMS monthly aggregated data
# with the 2021-Onward HMS monthly aggregated data
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
                "4th_Year_Paper_Data/HMS/2004-Onward/")
setwd(wd)

# Load packages
pacman::p_load(data.table)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Pre 2020 data path
pre_2020_path <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/",
                           "4th_Year_Paper_Data/HMS/2004-2020/Tobacco_Panelists_Purchases_2004-2020/",
                           "tobacco_panelists_purchases_monthly_CLEANED_2004-2020.rds")

# Post 2020 data path
post_2020_path <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/",
                            "4th_Year_Paper_Data/HMS/2021-Onward/Tobacco_Panelists_Purchases_2021-Onward/",
                            "tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.rds")


#############################
# Prepare data
#############################

# Load in full panel data
dt_pre <- readRDS(pre_2020_path)
dt_post <- readRDS(post_2020_path)

# Make columns identical across data sets
common_cols <- intersect(names(dt_pre), names(dt_post))
dt_pre <- dt_pre[, ..common_cols]
dt_post <- dt_post[, ..common_cols]

# Combine data
dt_panel <- rbindlist(list(dt_pre, dt_post), use.names = TRUE, fill = TRUE)

# Order and key by household, year, month
setkey(dt_panel, household_code, purchase_year, purchase_month)

# Indicator for when the flavor ban was implemented for those households
# in a state with a flavor ban

# 06 implemented in November 2019
# 25 implemented in November 2022
# 34 implemented in April 2020
# 36 implemented in May 2020
# 44 implemented in March 2020
# 49 implemented in July 2020
dt_panel[, flavor_ban := fifelse(
  (fips_state_code == 6  & purchase_year == 2019 & month(purchase_month) >= 11) |
    (fips_state_code == 25 & purchase_year == 2022 & month(purchase_month) >= 11) |
    (fips_state_code == 34 & purchase_year == 2020 & month(purchase_month) >= 4)  |
    (fips_state_code == 36 & purchase_year == 2020 & month(purchase_month) >= 5)  |
    (fips_state_code == 44 & purchase_year == 2020 & month(purchase_month) >= 3)  |
    (fips_state_code == 49 & purchase_year == 2020 & month(purchase_month) >= 7),
  1, 0
)]

# Get list of 20 most liberal states, including flavor ban implementing states
liberal <- c(6, 8, 9, 10, 15, 17, 23, 24, 25, 26,
             27, 32, 34, 35, 36, 41, 44, 49, 50, 51, 53)

# Filter out non-liberal states
dt <- dt_panel



library(fixest)

# Coerce outcome to numeric 0/1 if stored as factor/char
dt[, cig_num := as.integer(as.character(cig))]
dt[is.na(cig_num), cig_num := 0L]  # if any missing, treat as 0 (or drop instead)

# Collapse to state x month averages
dt_state <- dt[, .(
  mean_cig = sum(total_packs, na.rm = TRUE),   # outcome: share of households purchasing e-cigs
  flavor_ban = max(flavor_ban),             # state-level policy indicator
  household_count = .N
), by = .(fips_state_code, purchase_month)]

dt_state <- dt_state[!is.na(flavor_ban)]

# Add state-level adoption month
adopt_dt <- dt_state[, .(adopt_m = if (any(flavor_ban == 1))
  min(purchase_month[flavor_ban == 1]) else as.IDate(NA)),
  by = fips_state_code]
dt_state <- adopt_dt[dt_state, on = "fips_state_code"]


est <- feols(
  mean_cig ~ sunab(adopt_m, purchase_month, ref.p = -1) |
    fips_state_code + purchase_month,
  data = dt_state,
  cluster = ~ fips_state_code
)



summary(est)












