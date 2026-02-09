################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script creates the figure for flavored and unflavored
# e-cig purchase frequencies by household composition
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
pacman::p_load(data.table, ggplot2, grid, gridExtra)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Colors
blue <- "#56B4E9"
black <- "#000000"
orange <- "#D55E00"
grey <- "#666666"
pink <- "#CC79A7"
green <- "#3EB489"

# Image output directory
output_directory <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/",
                              "4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures/")
# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.rds")
dt <- readRDS(file_name)


#############################
# Functions
#############################

# Helper to build a 4-bar dataset: present vs absent for a given indicator
make_indicator_dt_plot <- function(ind_col, ind_label, dt_in) 
{
  tmp <- copy(dt_in)
  tmp[, group := fifelse(get(ind_col) == 1, paste0(ind_label, " present"), paste0(ind_label, " absent"))]
  out <- tmp[, .(
    flavored_rate   = mean(flavored_purchase),
    unflavored_rate = mean(unflavored_purchase)
  ), by = group]
  out
}


#############################
# E-cig purchase frequencies
# by household composition
#############################

# Household-month outcomes
hhm_ecig <- dt[, .(
  flavored_purchase   = as.integer(any((ecig == 1 | cig_ecig == 1) & flavored_ecig == 1, na.rm = TRUE)),
  unflavored_purchase = as.integer(any((ecig == 1 | cig_ecig == 1) & flavored_ecig == 0, na.rm = TRUE))
), by = .(household_code, purchase_month)]

# One row per household-month with the indicators attached
hhm_ind <- unique(dt[, .(
  household_code, purchase_month,
  teen_present,
  young_adult_present,
  teen_or_young_adult_present
)])
dt_merged <- merge(hhm_ind, hhm_ecig, by = c("household_code", "purchase_month"), all.x = TRUE)
dt_merged[is.na(flavored_purchase),   flavored_purchase := 0L]
dt_merged[is.na(unflavored_purchase), unflavored_purchase := 0L]

# Use the indicator dt_plot function 
dt_teen <- make_indicator_dt_plot("teen_present", "Teen", dt_merged)
dt_young_adult   <- make_indicator_dt_plot("young_adult_present", "Young adult", dt_merged)
dt_tya  <- make_indicator_dt_plot("teen_or_young_adult_present", "Teen or young adult", dt_merged)

# Combine into one plotting table with consistent ordering
dt_plot <- rbindlist(list(dt_teen, dt_young_adult, dt_tya), use.names = TRUE, fill = TRUE)
dt_plot_long <- melt(
  dt_plot,
  id.vars = "group",
  measure.vars = c("flavored_rate", "unflavored_rate"),
  variable.name = "type",
  value.name = "rate"
)
dt_plot_long[, type := factor(type, levels = c("flavored_rate", "unflavored_rate"))]
dt_plot_long[, group := factor(group, levels = c(
  "Teen present", "Teen absent",
  "Young adult present", "Young adult absent",
  "Teen or young adult present", "Teen or young adult absent"
))]

# Common y-axis
y_max <- dt_plot_long[, max(rate, na.rm = TRUE)]
y_lim <- c(0, min(1, y_max * 1.10))

# Plot
figure <- ggplot(dt_plot_long, aes(x = group, y = rate, fill = type)) +
  geom_col(
    color = black, linewidth = 1.2,
    position = position_dodge(width = 0.7),
    width = 0.6, alpha = 0.85
  ) +
  scale_fill_manual(
    values = c("flavored_rate" = orange, "unflavored_rate" = grey),
    labels = c("Flavored", "Unflavored"),
    name = NULL
  ) +
  scale_y_continuous(
    labels = scales::percent_format(),
    limits = y_lim,
    expand = c(0, 0)
  ) +
  labs(x = NULL, y = "E-Cigarette Purchase Frequency (Household-Month)") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position     = "bottom",
    legend.text         = element_text(size = 14, face = "bold"),
    panel.grid.minor.y  = element_blank(),
    panel.grid.major.y  = element_line(color = scales::alpha("grey85", 0.7)),
    panel.grid.minor.x  = element_blank(),
    panel.grid.major.x  = element_line(color = scales::alpha("grey85", 0.7), linewidth = 0.5),
    axis.line           = element_line(color = black, linewidth = 0.6),
    axis.ticks          = element_line(color = black, linewidth = 0.6),
    axis.ticks.length   = unit(-3, "pt"),
    axis.text.x         = element_text(size = 12, face = "bold", angle = 35, hjust = 1),
    axis.text.y         = element_text(size = 14, face = "bold"),
    axis.title.y        = element_text(size = 16, face = "bold")
  )
figure


#############################
# Write results to a file
#############################

# Write results
file_name <- paste0(output_directory, "/03_ECig_Flavored_Unflavored_Frequencies.txt")
fwrite(
  dt_plot_long,
  file = file_name,
  quote = FALSE
)

# Confirm results have been written
if (file.exists(file_name)) 
{
  cat("Figures have been written to", output_directory, "\n")
} else 
{
  cat("Error: Figures could not be written\n")
}




















