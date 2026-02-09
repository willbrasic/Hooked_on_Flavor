################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script creates a pie chart for the category level shares
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
# Pie chart of share of 
# category purchases by
# household-month
#############################

# Share of household-month purchases by category
number_of_household_months <- nrow(dt)

# Category shares
dt_shares <- data.table(
  share_outside   = nrow(dt[outside_option == 1]) / number_of_household_months,
  share_cig       = nrow(dt[cig == 1])            / number_of_household_months,
  share_ecig      = nrow(dt[ecig == 1])           / number_of_household_months,
  share_cig_ecig  = nrow(dt[cig_ecig == 1])       / number_of_household_months
)

# Long format
dt_shares_long <- melt(
  dt_shares,
  variable.name = "category",
  value.name    = "share"
)

# Clean labels
dt_shares_long[, category := fcase(
  category == "share_outside",  "Outside Option",
  category == "share_cig",      "Cigarettes",
  category == "share_ecig",     "E-Cigarettes",
  category == "share_cig_ecig", "Cigarettes & E-Cigarettes"
)]

# Explicit plotting order
dt_shares_long[, category := factor(
  category,
  levels = c(
    "Outside Option",
    "Cigarettes",
    "E-Cigarettes",
    "Cigarettes & E-Cigarettes"
  )
)]

# Named vector for legend percentages
shares_vec <- setNames(
  dt_shares_long$share,
  as.character(dt_shares_long$category)
)
legend_labels <- setNames(
  names(shares_vec),
  names(shares_vec)
)
legend_labels["E-Cigarettes"] <-
  sprintf("E-Cigarettes (%s)",
          scales::percent(shares_vec["E-Cigarettes"], accuracy = 0.1))
legend_labels["Cigarettes & E-Cigarettes"] <-
  sprintf("Cigarettes & E-Cigarettes (%s)",
          scales::percent(shares_vec["Cigarettes & E-Cigarettes"], accuracy = 0.1))

# Plot
figure <- ggplot(
  dt_shares_long,
  aes(x = "", y = share, fill = category)
) +
  geom_col(
    width = 1,
    color = black,
    linewidth = 1.2,
    alpha = 0.8
  ) +
  coord_polar(theta = "y") +
  geom_text(
    aes(
      label = ifelse(
        category %in% c("E-Cigarettes", "Cigarettes & E-Cigarettes"),
        "",
        scales::percent(share, accuracy = 0.1)
      )
    ),
    position = position_stack(vjust = 0.5),
    size = 6,
    fontface = "bold",
    color = black
  ) +
  scale_fill_manual(
    values = c(
      "Outside Option"              = orange,
      "Cigarettes"                  = blue,
      "E-Cigarettes"                = pink,
      "Cigarettes & E-Cigarettes"   = green
    ),
    labels = legend_labels,
    breaks = levels(dt_shares_long$category),
    limits = levels(dt_shares_long$category),
    name   = "Product Category"
  ) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid       = element_blank(),
    axis.text        = element_blank(),
    axis.title       = element_blank(),
    axis.ticks       = element_blank(),
    panel.background = element_rect(fill = NA, color = NA),
    plot.background  = element_rect(fill = NA, color = NA),
    legend.title     = element_text(size = 14, face = "bold"),
    legend.text      = element_text(size = 14, face = "bold")
  )
figure


#############################
# Write results to a file
#############################

# Write results to a file
file_name <- paste0(output_directory, "/A_01_Category_Shares_Plot.txt")
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


























