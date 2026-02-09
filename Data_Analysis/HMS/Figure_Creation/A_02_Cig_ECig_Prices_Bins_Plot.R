################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script creates a box plot for per-pack and per-mL prices by 
# consumption bins
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
# Box plots of per-pack and 
# per-mL of e-liquid prices
# by packs and mL purchased
#############################

# Quantity bins for cigs
breaks_packs <- c(0, 1, 2, 9, 10, 19, Inf)
labels_packs <- c("1", "2", "3 – 9", "10", "11 – 19", "20+")

# Quantity bins for e-cigs
breaks_mL <- c(0, 2, 4, 6, 8, 15, Inf)
labels_mL <- c("0 – 2", "2.01 – 4", "4.01 – 6", "6.01 – 8",
               "8.01 – 15", "15+")

# Prepare cig data for plot
dt_cig_box <- dt_cig[
  , packs_bin := cut(total_packs, breaks = breaks_packs, labels = labels_packs, right = TRUE, include.lowest = TRUE)
][, packs_bin := factor(packs_bin, levels = labels_packs)]

# Prepare e-cig data for plot
dt_ecig_box <- dt_ecig[
  , total_mL_bin := cut(total_mL, breaks = breaks_mL, labels = labels_mL, right = TRUE, lowest = TRUE)
][, total_mL_bin := factor(total_mL_bin, levels = labels_mL)]

# Common y-axis across both plots
y_min <- min(
  dt_cig_box[, per_pack_price_paid],
  dt_ecig_box[, per_mL_price_paid],
  na.rm = TRUE
)
y_max <- max(
  dt_cig_box[, per_pack_price_paid],
  dt_ecig_box[, per_mL_price_paid],
  na.rm = TRUE
)
y_min <- floor(y_min)
y_max <- ceiling(y_max)
y_breaks <- seq(0, ceiling(y_max), by = 4)

# Plot per-pack price paid by pack quantity bins
gg_cig_box <- ggplot(dt_cig_box, aes(x = packs_bin, y = per_pack_price_paid)) +
  geom_boxplot(
    outlier.shape = NA,              
    width = 0.6,                    
    color = grey, 
    fill = scales::alpha(blue, 0.8),
    linewidth = 1.2,             
    fatten = 1.2  
    ) +
  stat_summary(
    fun = median,
    geom = "crossbar",
    width = 0.6,
    color = black,          
    linewidth = 1
    ) +
  stat_summary(
    fun.min = min, 
    fun.max = max, 
    geom = "errorbar", 
    width = 0.6, 
    color = grey, 
    linewidth = 1.2
    ) +
  labs(
    title = "Panel (a)",
    x = "Packs of Cigarettes Purchased", 
    y = "Per-Pack Price Paid ($)"
    ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(limits = c(y_min, y_max), breaks = y_breaks) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_line(color = scales::alpha("grey85", 0.7)),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.x = element_line(color = scales::alpha("grey85", 0.7), linewidth = 0.5),
    axis.line = element_line(color = black, linewidth = 0.6),
    axis.ticks = element_line(color = black, linewidth = 0.6),
    axis.ticks.length = unit(-3, "pt"),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5, vjust = 1.5), 
    axis.text.x = element_text(size = 14, face = "bold", angle = 45, hjust = 0.5, vjust = 0.5),
    axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold"),
    plot.margin = margin(t = 5, r = 5, b = 20, l = 5)
    )

# Plot per-mL of e-liquid price paid by mL quantity bins
gg_ecig_box <- ggplot(dt_ecig_box, aes(x = total_mL_bin, y = per_mL_price_paid)) +
  geom_boxplot(
    outlier.shape = NA,              
    width = 0.6,                    
    color = grey, 
    fill = scales::alpha(pink, 0.8),
    linewidth = 1.2,             
    fatten = 1.2  
    ) +
  stat_summary(
    fun = median,
    geom = "crossbar",
    width = 0.6,
    color = black,          
    linewidth = 1
    ) +
  stat_summary(
    fun.min = min, 
    fun.max = max, 
    geom = "errorbar", 
    width = 0.6, 
    color = grey, 
    linewidth = 1.2
    ) +
  labs(
    title = "Panel (b)",
    x = "mL of E-Liquid Purchased", 
    y = "Per-mL Price Paid ($)"
    ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(limits = c(y_min, y_max), breaks = y_breaks) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_line(color = scales::alpha("grey85", 0.7)),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.x = element_line(color = scales::alpha("grey85", 0.7), linewidth = 0.5),
    axis.line = element_line(color = black, linewidth = 0.6),
    axis.ticks = element_line(color = black, linewidth = 0.6),
    axis.ticks.length = unit(-3, "pt"),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5, vjust = 1.5), 
    axis.text.x = element_text(size = 14, face = "bold", angle = 45, hjust = 0.5, vjust = 0.5),
    axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold")
    )

# Combine plots into single figure
figure <- grid.arrange(gg_cig_box, gg_ecig_box, ncol = 2)

# Write figure to output directory 
figure_name <- "/Per-Pack_Per-mL_Prices_Box.png"
ggsave(paste0(output_directory, figure_name), plot = figure, width = 14, height = 6, dpi = 300)

# Confirm results have been written to a file
if (file.exists(paste0(output_directory, figure_name))) 
{
  cat("Figure has been written to", paste0(output_directory, figure_name), "\n")
} else 
{
  cat("Error: Figure could not be written\n")
}












