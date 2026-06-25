################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# Five-number summaries (min, Q1, median, Q3, max) of per-pack and per-mL
# prices by consumption bin, exported as comma-separated .txt files for
# pgfplots boxplots in the paper's LaTeX source.
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

# Output directory for pgfplots data files
output_directory <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/",
                              "4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures/")

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.rds")
dt <- readRDS(file_name)


#############################
# Prepare cigarette data
#############################

# Filter to cigarette-only purchase months (positive packs, zero e-liquid)
dt_cig <- dt[total_packs > 0 & total_mL == 0]

# Quantity bins for cigarettes: 1-9, 10-19, 20-29, 30-39, 40+
breaks_packs <- c(0, 9, 19, 29, 39, Inf)
labels_packs <- c("1 – 9", "10 – 19", "20 – 29", "30 – 39", "40+")

# Assign bins
dt_cig[, packs_bin := cut(total_packs,
                           breaks = breaks_packs,
                           labels = labels_packs,
                           right = TRUE,
                           include.lowest = TRUE)]
dt_cig[, packs_bin := factor(packs_bin, levels = labels_packs)]

# Compute five-number summary by bin
dt_cig_summary <- dt_cig[, .(
  lower_whisker  = min(per_pack_price_paid, na.rm = TRUE),
  lower_quartile = quantile(per_pack_price_paid, 0.25, na.rm = TRUE),
  median         = median(per_pack_price_paid, na.rm = TRUE),
  upper_quartile = quantile(per_pack_price_paid, 0.75, na.rm = TRUE),
  upper_whisker  = max(per_pack_price_paid, na.rm = TRUE)
), keyby = packs_bin]

# Add numeric x-position for pgfplots (1, 2, ..., number of bins)
dt_cig_summary[, x := .I]

cat("Cigarette price summary by bin:\n")
print(dt_cig_summary)


#############################
# Prepare e-cigarette data
#############################

# Filter to e-cig-only purchase months (positive e-liquid, zero packs)
dt_ecig <- dt[total_mL > 0 & total_packs == 0]

# Quantity bins for e-cigarettes: 0-5, 5-15, 15-30, 30-50, 50+
breaks_mL <- c(0, 5, 15, 30, 50, Inf)
labels_mL <- c("0 – 5", "5 – 15", "15 – 30", "30 – 50", "50+")

# Assign bins
dt_ecig[, total_mL_bin := cut(total_mL,
                               breaks = breaks_mL,
                               labels = labels_mL,
                               right = TRUE,
                               include.lowest = TRUE)]
dt_ecig[, total_mL_bin := factor(total_mL_bin, levels = labels_mL)]

# Compute five-number summary by bin
dt_ecig_summary <- dt_ecig[, .(
  lower_whisker  = min(per_mL_price_paid, na.rm = TRUE),
  lower_quartile = quantile(per_mL_price_paid, 0.25, na.rm = TRUE),
  median         = median(per_mL_price_paid, na.rm = TRUE),
  upper_quartile = quantile(per_mL_price_paid, 0.75, na.rm = TRUE),
  upper_whisker  = max(per_mL_price_paid, na.rm = TRUE)
), keyby = total_mL_bin]

# Add numeric x-position for pgfplots
dt_ecig_summary[, x := .I]

cat("\nE-cigarette price summary by bin:\n")
print(dt_ecig_summary)


#############################
# Write summary statistics
# to .txt files for pgfplots
#############################

# Write cigarette summary
fwrite(dt_cig_summary[, .(x, lower_whisker, lower_quartile, median, upper_quartile, upper_whisker)],
       file = paste0(output_directory, "A_02_Cig_Prices_Bins_Box.txt"),
       sep = ",")

# Write e-cigarette summary
fwrite(dt_ecig_summary[, .(x, lower_whisker, lower_quartile, median, upper_quartile, upper_whisker)],
       file = paste0(output_directory, "A_02_ECig_Prices_Bins_Box.txt"),
       sep = ",")

# Confirm results have been written
if (file.exists(paste0(output_directory, "A_02_Cig_Prices_Bins_Box.txt")))
{
  cat("\nCigarette summary written to", paste0(output_directory, "A_02_Cig_Prices_Bins_Box.txt"), "\n")
} else
{
  cat("\nError: Cigarette summary could not be written\n")
}

if (file.exists(paste0(output_directory, "A_02_ECig_Prices_Bins_Box.txt")))
{
  cat("E-cigarette summary written to", paste0(output_directory, "A_02_ECig_Prices_Bins_Box.txt"), "\n")
} else
{
  cat("Error: E-cigarette summary could not be written\n")
}
