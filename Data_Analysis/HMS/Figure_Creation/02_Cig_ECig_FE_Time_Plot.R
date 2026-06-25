################################################################################
# William Brasic 
# The Unvarersity of Arizona
# wbrasic97@gmail.com
# September 2025
#
# Within-household fixed-effects plots of nicotine intake over time for
# cigarette and e-cigarette users.
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
pacman::p_load(data.table, sandwich, lmtest, fixest, broom)

# Suppress scientific notation
options(scipen = 999)

# Print limit
options(max.print = 999999)

# Data table print limit
options(datatable.print.nrows = 999)

# Image output directory
output_directory <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA",
                              "4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures/")

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt <- fread(file_name)


#############################
# Fixed effects plot
# over time
############################# 

# Use balanced panel to avoid average household purchases increasing over time
# b/c higher frequency smokers join in later
dt_filtered <- dt[
  , if (length(unique(purchase_year)) == length(unique(dt$purchase_year))) .SD,
  by = household_code
]

# OLS of nicotine consumed from cigs for those who purchased cigs
ols_cig_month <- feols(
  log(cig_nicotine_mg_consumed) ~ i(purchase_month) + real_per_pack_price_paid * total_packs + 
    household_income + household_size |
    household_code,
  data = dt_filtered[cig == 1 | cig_ecig == 1],
  panel.id = ~ household_code + purchase_month,
  vcov = ~ household_code
)

# Results table
dt_cig <- as.data.table(tidy(ols_cig_month, conf.int = TRUE))
dt_cig <- dt_cig[grepl("^purchase_month::", term),
                 .(
                   month = sub("^purchase_month::", "", term),
                   estimate,
                   conf.low,
                   conf.high
                 )]

# OLS of nicotine consumed from e-cigs for those who purchased e-cigs
ols_ecig_month <- feols(
  log(ecig_nicotine_mg_consumed) ~ i(purchase_month) + real_per_mL_price_paid * total_mL +
    household_income + household_size |
    household_code,
  data = dt_filtered[ecig == 1 | cig_ecig == 1],
  panel.id = ~ household_code + purchase_month,
  vcov = ~ household_code
)

# Results table
dt_ecig <- as.data.table(tidy(ols_ecig_month, conf.int = TRUE))
dt_ecig <- dt_ecig[grepl("^purchase_month::", term),
                   .(
                     month = sub("^purchase_month::", "", term),
                     estimate,
                     conf.low,
                     conf.high
                   )]


#############################
# Write results to a file
#############################

# Write cig results
file_name_cig <- paste0(output_directory, "/02_Cig_FE_Time_Plot.txt")
fwrite(
  dt_cig,
  file = file_name_cig,
  sep = " ",
  quote = FALSE
)

# Write e-cig results
file_name_ecig <- paste0(output_directory, "/02_ECig_FE_Time_Plot.txt")
fwrite(
  dt_ecig,
  file = file_name_ecig,
  sep = " ",
  quote = FALSE
)

# Confirm results have been written
if (file.exists(file_name_cig) & file.exists(file_name_ecig)) 
{
  cat("Figures have been written to", output_directory, "\n")
} else 
{
  cat("Error: Figures could not be written\n")
}
















