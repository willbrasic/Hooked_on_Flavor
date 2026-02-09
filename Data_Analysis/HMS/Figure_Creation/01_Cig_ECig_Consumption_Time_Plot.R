################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script creates the figure for inflation-adjusted 
# median monthly per-pack and per-mL prices.
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
# Plot median of purchased
# packs and mL of e-liquid
# by month
#############################

# Prepare data for plots
dt_cig_monthly  <- dt_cig[, .(total_packs_all = sum(total_packs, na.rm = TRUE)), 
                          by = purchase_month][order(purchase_month)]
dt_ecig_monthly <- dt_ecig[, .(total_mL_all = sum(total_mL, na.rm = TRUE)), 
                           by = purchase_month][order(purchase_month)]
dt_both <- merge(dt_cig_monthly, dt_ecig_monthly, by = "purchase_month", all = TRUE)
dt_both[is.na(total_packs_all), total_packs_all := 0]
dt_both[is.na(total_mL_all),   total_mL_all := 0]
range_start <- as.Date(cut(min(dt_both[, purchase_month], na.rm = TRUE), "month"))
range_end <- as.Date(cut(max(dt_both[, purchase_month], na.rm = TRUE), "month"))
breaks_3mo <- seq(range_start, range_end, by = "3 months")

# Scale e-cig series to cigarette axis: y* = a * total_mL_all + b
r1 <- range(dt_both$total_packs_all, na.rm = TRUE)
r2 <- range(dt_both$total_mL_all,   na.rm = TRUE)
if (diff(r2) == 0) 
{
  a <- 1
  b <- r1[1] - a * r2[1]
} else 
{
  a <- diff(r1) / diff(r2)
  b <- r1[1] - a * r2[1]
}

# Plot cig and ecig data on single plot
figure <- ggplot(dt_both, aes(x = purchase_month)) +
  geom_line(aes(y = total_packs_all, color = "Cigarettes"), linewidth = 2, alpha = 0.8) +
  geom_point(aes(y = total_packs_all, color = "Cigarettes", shape = "Cigarettes"),
             size = 4.5, stroke = 1, fill = black) +
  geom_line(aes(y = a * total_mL_all + b, color = "E-Cigarettes"), linewidth = 2, alpha = 0.8) +
  geom_point(aes(y = a * total_mL_all + b, color = "E-Cigarettes", shape = "E-Cigarettes"),
             size = 4.5, stroke = 1, fill = grey) +
  labs(
    x = "Month",
    color = "Product Type",
    shape = "Product Type"
  ) +
  scale_color_manual(
    name = NULL,
    values = c("Cigarettes" = blue,  
               "E-Cigarettes" = pink) 
  ) +
  scale_shape_manual(
    name = NULL,
    values = c("Cigarettes" = 21, "E-Cigarettes" = 21)
  ) +
  guides(
    color = guide_legend(title.position = "bottom", title.hjust = 0.5),
    shape = guide_legend(title.position = "bottom", title.hjust = 0.5)
  ) +
  scale_x_date(
    breaks = breaks_3mo,
    labels = date_format("%b\n%Y"),
    minor_breaks = NULL,
    expand = c(0.01, 0.01)
  ) +
  scale_y_continuous(
    name = "Total Packs Purchased",
    labels = scales::comma,
    limits = c(min(dt_both$total_packs_all, na.rm = TRUE) * 0.97,
               max(dt_both$total_packs_all, na.rm = TRUE) * 1.03),
    expand = c(0, 0),
    sec.axis = sec_axis(~ (. - b) / a, name = "Total mL of E-Liquid Purchased", labels = scales::comma)
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12, face = "bold"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_line(color = scales::alpha("grey85", 0.7)),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.x = element_line(color = scales::alpha("grey85", 0.7), linewidth = 0.5),
    axis.line = element_line(color = black, linewidth = 0.6),
    axis.ticks = element_line(color = black, linewidth = 0.6),
    axis.ticks.length = unit(-3, "pt"),
    axis.text.x = element_text(size = 12, face = "bold", angle = 45, hjust = 1),
    axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y.left  = element_text(size = 16, face = "bold"),
    axis.title.y.right = element_text(size = 16, face = "bold")
  )

# Show the figure
figure


#############################
# Write results to a file
#############################

# Write results to a file
file_name <- paste0(output_directory, "/01_Cig_ECig_Consumption_Time_Plot.txt")
fwrite(
  dt_both,
  file = file_name,
  quote = FALSE
)

# Confirm results have been written
if (file.exists(file_name)) 
{
  cat("Results have been written to", file_name, "\n")
} else 
{
  cat("Error: Figures could not be written\n")
}























