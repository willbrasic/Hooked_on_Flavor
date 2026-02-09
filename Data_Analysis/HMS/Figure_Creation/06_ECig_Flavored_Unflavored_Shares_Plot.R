################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script creates a pie chart for flavored and unflavored e-cigarette
# purchase shares
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

# Filter by product category
dt_ecig <- dt[ecig == 1 | cig_ecig == 1]


#############################
# Pie chart of e-cig shares 
# by flavored or not
#############################

# Number of e-cig purchases
number_of_ecig_purchases <- nrow(dt_ecig)
number_of_flavored_ecig_purchases <- nrow(dt_ecig[flavored_ecig == 1])
number_of_unflavored_ecig_purchases <- nrow(dt_ecig[original_ecig == 1])

# Shares based on flavored vs unflavored e-cig purchases
dt_shares_long <- data.table(
  segment = c("Flavored", "Unflavored"),
  share   = c(
    number_of_flavored_ecig_purchases / number_of_ecig_purchases,
    number_of_unflavored_ecig_purchases / number_of_ecig_purchases
  )
)

# Pie chart of share of e-cig purchases by flavor status
figure <- ggplot(dt_shares_long, aes(x = "", y = share, fill = segment)) +
  geom_col(width = 1, color = "black", linewidth = 1.2, alpha = 0.8) +
  coord_polar(theta = "y") +
  geom_text(
    aes(label = scales::percent(share, accuracy = 0.1)),
    position = position_stack(vjust = 0.5),
    size = 6, fontface = "bold", color = "black"
  ) +
  scale_fill_manual(
    values = c(
      "Flavored" = orange,
      "Unflavored" = grey
    )
  ) +
  labs(
    x = NULL,
    y = NULL,
    fill = "E-Cig Type"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    panel.background = element_rect(fill = NA, color = NA),
    plot.background = element_rect(fill = NA, color = NA),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5, vjust = 1.5),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text  = element_text(size = 16, face = "bold")
  )
figure


#############################
# Write results to a file
#############################

# Write results to a file
file_name <- paste0(output_directory, "/06_ECig_Flavored_Unflavored_Shares_Plot.txt")
fwrite(
  dt_shares_long,
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






