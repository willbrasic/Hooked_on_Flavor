################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# March 2026
#
# Streak persistence curve figure for model validation: actual vs predicted
# continuation rates by streak length for cigarettes and e-cigarettes.
################################################################################


#############################
# Preliminaries
#############################

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# Load packages
pacman::p_load(data.table, ggplot2, gridExtra, scales)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Colors
blue <- "#56B4E9"
black <- "#000000"
orange <- "#D55E00"
grey <- "#666666"
pink <- "#CC79A7"
green <- "#3EB489"

# Image output directory
output_directory <- "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures"


#############################
# Load Data
#############################

# Load streak persistence CSV from mixture model validation
dt <- fread("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Mixture_Validation_Streak_Persistence.csv")

# Convert streak_length to numeric (replace "12+" with 12)
dt[, streak_num := as.numeric(gsub("\\+", "", streak_length))]


#############################
# Write figure data for
# pgfplots (space-separated)
#############################

# Cigarettes
dt_cig <- dt[product == "Cigarettes", .(streak = streak_num, actual = actual_rate, predicted = predicted_rate)]
file_name <- paste0(output_directory, "08_Streak_Persistence_Cig.txt")
fwrite(dt_cig, file = file_name, quote = FALSE, sep = " ")

# E-Cigarettes
dt_ecig <- dt[product == "E-Cigarettes", .(streak = streak_num, actual = actual_rate, predicted = predicted_rate)]
file_name <- paste0(output_directory, "08_Streak_Persistence_ECig.txt")
fwrite(dt_ecig, file = file_name, quote = FALSE, sep = " ")

# Confirm
if (file.exists(paste0(output_directory, "08_Streak_Persistence_Cig.txt")) &
    file.exists(paste0(output_directory, "08_Streak_Persistence_ECig.txt")))
{
  cat("Streak persistence figure data written successfully.\n")
} else
{
  cat("Error: Figure data could not be written.\n")
}
