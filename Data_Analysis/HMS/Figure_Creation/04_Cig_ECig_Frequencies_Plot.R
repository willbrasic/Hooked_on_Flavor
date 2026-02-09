################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script creates the figure for the monthly purchase frequencies 
# of packs of cigarettes and mL of e-liquid
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
pacman::p_load(data.table, ggplot2, gridExtra, scales)

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

# Get category specific purchase data tables
dt_cig <- dt[cig == 1 | cig_ecig == 1]
dt_ecig <- dt[ecig == 1 | cig_ecig == 1]


#############################
# Plot of purchase 
# frequencies for packs
# and mL of e-liquid
#############################

# Quantity bins for e-cigs
breaks_mL <- c(0, 5, 10, 15, 20, 30, 50, Inf)
labels_mL <- c("0 – 5", "5.01 – 10", "10.01 – 15", "15.01 – 20",
               "20.01 – 30", "30.01 – 50", "50+")

# Prepare cig data for plot
dt_cig_plot <- dt_cig[, .(purchases = .N), by = total_packs][order(total_packs)]
dt_cig_plot <- rbind(
  dt_cig_plot[total_packs <= 40],
  dt_cig_plot[total_packs > 40,
              .(total_packs = 41, purchases = sum(purchases))]
)
dt_cig_plot[, frequency := purchases / sum(purchases)]
dt_cig_plot[, total_packs_label := ifelse(total_packs == 41, "41+", as.character(total_packs))]
dt_cig_plot[, total_packs_plot := ifelse(total_packs == 41, total_packs + 1, total_packs)]

# Prepare e-cig data for plot 
types <- c("Original", "Flavored")
dt_ecig_long <- melt(
  dt_ecig,
  measure.vars = c("total_original_mL", "total_flavored_mL"),
  variable.name = "type_raw",
  value.name = "mL"
)[!is.na(mL) & mL > 0]
dt_ecig_long[, type := fifelse(type_raw == "total_original_mL", "Original", "Flavored")]
dt_ecig_long[, mL_bin := cut(
  mL,
  breaks = breaks_mL,
  labels = labels_mL,
  right = TRUE,
  include.lowest = FALSE
)]
dt_ecig_plot <- dt_ecig_long[, .(purchases = .N), by = .(type, mL_bin)]
all_combos <- CJ(
  type = factor(types, levels = types),
  mL_bin = factor(labels_mL, levels = labels_mL)
)
dt_ecig_plot <- merge(all_combos, dt_ecig_plot, by = c("type", "mL_bin"), all.x = TRUE)
dt_ecig_plot[is.na(purchases), purchases := 0]

# Frequencies over all e-cig purchases (Original + Flavored)
dt_ecig_plot[, frequency := purchases / sum(purchases)]
max_freq_ecig <- max(dt_ecig_plot[, frequency])

# Plot frequencies of cigarette pack purchases
max_x <- max(dt_cig_plot[, total_packs_plot])
breaks_x <- sort(unique(c(1, seq(5, 40, by = 5), 42)))  
gg_cig_freg <- ggplot(dt_cig_plot, aes(x = total_packs_plot, y = frequency)) +
  geom_vline(
    xintercept = breaks_x[breaks_x != 42],
    color = scales::alpha("grey85", 0.7),
    linewidth = 0.5
  ) +
  geom_col(
    fill = blue, 
    color = black, 
    linewidth = 1.2, 
    alpha = 0.8
  ) +
  labs(
    title = "Panel (a)",
    x = "Packs of Cigarettes Purchased", 
    y = "Frequency") +
  scale_y_continuous(
    labels = scales::percent_format(),
    expand = c(0, 0),
    limits = c(0, max(dt_ecig_plot[, frequency]) * 1.1)
  ) +
  scale_x_continuous(
    limits = c(0, max_x + 1),
    breaks = c(1, 5, 10, 15, 20, 25, 30, 35, 40, 42),
    labels = c("1", "5", "10", "15", "20", "25", "30", "35", "40", "41+"),
    expand = c(0, 0)
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_blank(),  
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_line(color = scales::alpha("grey85", 0.7)),
    panel.grid.minor.y = element_blank(),
    axis.line = element_line(color = black, linewidth = 0.6),
    axis.ticks = element_line(color = black, linewidth = 0.6),
    axis.ticks.length = unit(-3, "pt"),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5, vjust = 1.5),
    axis.text.x = element_text(size = 14, face = "bold", angle = 45, hjust = 0.5, vjust = 0.5),
    axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold"),
    plot.margin = margin(t = 5, r = 5, b = 40, l = 5)
  )

# Plot frequencies of mL of e-liquid purchased
gg_ecig_freg <- ggplot(dt_ecig_plot, aes(x = mL_bin, y = frequency, fill = type)) +
  geom_col(
    position = position_dodge(width = 0.85),
    color = black,
    linewidth = 1.2,
    alpha = 0.85,
    width = 0.8
  ) +
  labs(
    title = "Panel (b)",
    x = "mL of E-Liquid Purchased",
    y = "Frequency",
    fill = NULL
  ) +
  scale_y_continuous(
    labels = scales::percent_format(),
    expand = c(0, 0),
    limits = c(0, max_freq_ecig * 1.1)
  ) +
  scale_fill_manual(values = c("Original" = grey, "Flavored" = pink)) +
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
    legend.position = "none"
  )

# Combine plots into single figure
figure <- grid.arrange(gg_cig_freg, gg_ecig_freg, nrow = 2)


#############################
# Write results to a file
#############################

# Write cig results
file_name_cig <- paste0(output_directory, "/04_Cig_Frequencies_Plot.txt")
fwrite(
  dt_cig_plot,
  file = file_name_cig,
  sep = " ",
  quote = FALSE
)

# Write e-cig results
file_name_ecig <- paste0(output_directory, "/04_ECig_Frequencies_Plot.txt")
fwrite(
  dt_ecig_plot,
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











