################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Figure for FDA authorized vs non-FDA authorized e-cig purchase frequencies by year
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

# Load in monthly panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.rds")
dt <- readRDS(file_name)


#############################
# FDA authorized e-cig
# purchase frequencies by year,
# flavor, and TYA presence
#############################

# Subset to e-cig purchase months
dt_ecig <- dt[ecig == 1 | cig_ecig == 1]

# Subset to flavored e-cig purchase months only
dt_ecig_flav <- dt_ecig[flavored_ecig == 1]

# Compute FDA authorized and non-FDA shares by year and TYA
dt_plot <- dt_ecig_flav[, .(
  fda_authorized_rate = mean(fda_authorized_ecig == 1),
  non_fda_rate = mean(fda_authorized_ecig == 0)
), keyby = .(teen_or_young_adult_present, purchase_year)]

# Reshape to long format for ggplot
dt_plot_long <- melt(
  dt_plot,
  id.vars = c("teen_or_young_adult_present", "purchase_year"),
  measure.vars = c("fda_authorized_rate", "non_fda_rate"),
  variable.name = "type",
  value.name = "rate"
)
dt_plot_long[, type := factor(type, levels = c("fda_authorized_rate", "non_fda_rate"))]
dt_plot_long[, purchase_year := factor(purchase_year)]

# Facet label
dt_plot_long[, tya_label := fifelse(teen_or_young_adult_present == 1,
                                    "TYA Present", "No TYA")]
dt_plot_long[, tya_label := factor(tya_label, levels = c("No TYA", "TYA Present"))]

# Common y-axis
y_lim <- c(0, 0.80)

# Plot
figure <- ggplot(dt_plot_long, aes(x = purchase_year, y = rate, fill = type)) +
  geom_col(
    color = black, linewidth = 1.2,
    position = position_dodge(width = 0.7),
    width = 0.6, alpha = 0.85
  ) +
  geom_text(
    aes(label = scales::percent(rate, accuracy = 0.1)),
    position = position_dodge(width = 0.7),
    vjust = -0.5, size = 3.5, fontface = "bold"
  ) +
  facet_wrap(~ tya_label) +
  scale_fill_manual(
    values = c("fda_authorized_rate" = blue, "non_fda_rate" = grey),
    labels = c("FDA Authorized", "Non-FDA Authorized"),
    name = NULL
  ) +
  scale_y_continuous(
    labels = scales::percent_format(),
    limits = y_lim,
    expand = c(0, 0)
  ) +
  labs(x = NULL, y = "Share of Flavored E-Cigarette Purchase Months") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position     = "bottom",
    legend.text         = element_text(size = 14, face = "bold"),
    strip.text          = element_text(size = 14, face = "bold"),
    panel.grid.minor.y  = element_blank(),
    panel.grid.major.y  = element_line(color = scales::alpha("grey85", 0.7)),
    panel.grid.minor.x  = element_blank(),
    panel.grid.major.x  = element_line(color = scales::alpha("grey85", 0.7), linewidth = 0.5),
    axis.line           = element_line(color = black, linewidth = 0.6),
    axis.ticks          = element_line(color = black, linewidth = 0.6),
    axis.ticks.length   = unit(-3, "pt"),
    axis.text.x         = element_text(size = 14, face = "bold"),
    axis.text.y         = element_text(size = 14, face = "bold"),
    axis.title.y        = element_text(size = 16, face = "bold")
  )
figure


#############################
# Write results to a file
#############################

# Write results
file_name <- paste0(output_directory, "/07_ECig_FDA_Authorized_Frequencies.txt")
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




































