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
pacman::p_load(data.table)

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


#############################
# Monthly panel   
############################# 

# Median monthly consumption
dt_monthly_cig[, median(total_packs, na.rm = TRUE)]
dt_monthly_ecig[, median(total_mL, na.rm = TRUE)]

























