################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# Figure for inflation-adjusted median monthly per-pack and per-mL prices.
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

# Suppress scientific notation
options(scipen = 999)

# Print limit
options(max.print = 999999)

# Data table print limit
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

# Category-specific purchase data
dt_cig <- dt[cig == 1 | cig_ecig == 1]
dt_ecig <- dt[ecig == 1 | cig_ecig == 1]


#############################
# Plot median of real 
# per-unit prices for cigs 
# and e-cigs by month
#############################

# Prepare cig data for plot
dt_cig_prices <- dt_cig[, .(real_median_price = median(real_per_pack_price_paid, na.rm = TRUE)), 
                        by = purchase_month]

# Prepare e-cig data for plot
dt_ecig_prices <- dt_ecig[, .(real_median_price = median(real_per_mL_price_paid, na.rm = TRUE)), 
                          by = purchase_month]
# Merge cig and e-cig price data
dt_both <- merge(
  dt_cig_prices[,   .(purchase_month, cig = real_median_price)],
  dt_ecig_prices[, .(purchase_month, ecig = real_median_price)],
  by = "purchase_month", all = TRUE
)

# x-ticks are every months
range_start <- as.Date(cut(min(dt_both[, purchase_month], na.rm = TRUE), "month"))
range_end   <- as.Date(cut(max(dt_both[, purchase_month], na.rm = TRUE), "month"))
breaks_3mo  <- seq(range_start, range_end, by = "3 months")

# Scale ecig to the cig axis: y* = a*ecig + b
r1 <- range(dt_both[, cig],  na.rm = TRUE)
r2 <- range(dt_both[, ecig], na.rm = TRUE)
a <- diff(r1) / diff(r2)
b <- r1[1] - a * r2[1]

# Plot cig and e-cig per-unit prices on a single plot 
figure <- ggplot(dt_both, aes(x = purchase_month)) +
  geom_line(aes(y = cig, color = "Cigarettes"), linewidth = 2, alpha = 0.8) +
  geom_point(aes(y = cig, color = "Cigarettes", shape = "Cigarettes"), size = 4.5, stroke = 1, fill = black) +
  geom_line(aes(y = a * ecig + b, color = "E-Cigarettes"), linewidth = 2, alpha = 0.8) +
  geom_point(aes(y = a * ecig + b, color = "E-Cigarettes", shape = "E-Cigarettes"), size = 4.5, stroke = 1, fill = grey) +
  labs(
    x = "Month",
    color = "Product Type",
    shape = "Product Type"
  ) +
  scale_color_manual(
    name = NULL,
    values = c("Cigarettes" = blue, "E-Cigarettes" = pink)
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
    name = "Median Per-Pack Price Paid ($)",
    limits = c(min(dt_both[, cig], na.rm = TRUE) * 0.97,
               max(dt_both[, cig], na.rm = TRUE) * 1.03),
    expand = c(0, 0),
    sec.axis = sec_axis(~ (. - b) / a, name = "Median Per-mL Price Paid ($)")
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12, face = "bold"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_line(color = alpha("grey85", 0.7)),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.x = element_line(color = alpha("grey85", 0.7), linewidth = 0.5),
    axis.line = element_line(color = black, linewidth = 0.6),
    axis.ticks = element_line(color = black, linewidth = 0.6),
    axis.ticks.length = unit(-3, "pt"),
    axis.text.x = element_text(size = 12, face = "bold", angle = 45, hjust = 1),
    axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y.left  = element_text(size = 16, face = "bold"),
    axis.title.y.right = element_text(size = 16, face = "bold")
  )
figure


#############################
# Write results to a file
#############################

# Write results to a file
file_name <- paste0(output_directory, "/05_Cig_ECig_Prices_Time_Plot.txt")
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
















































