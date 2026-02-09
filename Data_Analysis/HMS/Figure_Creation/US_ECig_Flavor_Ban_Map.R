################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script plots a map of the contiguous U.S. highlighting states that
# have banned flavored e-cigarettes and the month of implementation.
################################################################################


#############################
# Preliminaries   
############################# 

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Load packages
pacman::p_load(data.table, tigris, sf, ggplot2)

# Colors
blue <- "#56B4E9"
orange <- "#D55E00"
grey <- "#666666"
pink <- "#CC79A7"
green <- "#3EB489"
maroon <- "#800000"

# Image output directory
output_directory <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/",
                              "4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures/")

#############################
# Plot
#############################

# Function to plot the map
plot_flavor_bans <- function(bans_df, insets = TRUE) 
{
  # Normalize bans input with abbreviations
  abbr_map <- data.table(
    abbr = c(state.abb, "DC", "D.C."),
    name = c(state.name, rep("District of Columbia", 2))
  )
  
  bans_dt <- as.data.table(bans_df)
  bans_dt[, state := as.character(state)]
  bans_dt[, state := trimws(state)]
  bans_dt[, date  := as.IDate(as.Date(date))]
  
  # Left-join on abbreviations
  bans_dt <- merge(
    bans_dt, abbr_map,
    by.x = "state", by.y = "abbr",
    all.x = TRUE, sort = FALSE
  )
  bans_dt[, state_std := ifelse(!is.na(name), name, state)]
  bans_dt[, month_lab_chr := format(as.Date(date), "%b %Y")]
  setorder(bans_dt, date)
  bans_dt[, month_lab := factor(month_lab_chr, levels = unique(month_lab_chr))]
  
  # Map geometry
  options(tigris_use_cache = TRUE)
  suppressMessages({
    st <- tigris::states(cb = TRUE, year = 2023, class = "sf")
    if (isTRUE(insets)) st <- tigris::shift_geometry(st)
  })
  
  keep_states <- c(state.name, "District of Columbia")
  st <- st[st$NAME %in% keep_states, ]
  st$state_key <- toupper(st$NAME)
  
  # Prep bans for merge
  bans_dt2 <- bans_dt[, .(state_key = toupper(state_std), month_lab)]
  
  # Merge bans onto shapes
  st_bans <- merge(st, as.data.frame(bans_dt2), by = "state_key", all.x = TRUE, sort = FALSE)
  
  # Color mapping
  base_colors <- c(blue, orange, grey, pink, green, maroon)
  lvls <- levels(st_bans$month_lab)
  named_colors <- stats::setNames(rep_len(base_colors, length(lvls)), lvls)
  
  # Plot
  bb <- sf::st_bbox(st)
  ggplot() +
    geom_sf(
      data = subset(st_bans, is.na(month_lab)),
      fill = "grey85", color = "white", linewidth = 0.3
    ) +
    geom_sf(
      data = subset(st_bans, !is.na(month_lab)),
      aes(fill = month_lab),
      color = "white", linewidth = 0.3
    ) +
    coord_sf(
      xlim = c(bb["xmin"], bb["xmax"]),
      ylim = c(bb["ymin"], bb["ymax"]),
      expand = FALSE
    ) +
    scale_fill_manual(values = named_colors, name = NULL) +  # no legend title
    theme_minimal(base_size = 12) +
    theme(
      legend.text = element_text(size = 16, face = "bold"),
      panel.grid = element_blank(),
      axis.text  = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      legend.position = "right"
    )
}

# States with ban and the month of implementation
state_bans <- data.table(
  state = c("Massachusetts","New Jersey","New York","Rhode Island","Utah","California"),
  date  = c("2019-11-01","2020-04-01","2020-05-18","2020-03-01","2020-07-01","2022-12-01")
)

# Plot figure
figure <- plot_flavor_bans(state_bans)

# Write figure to output directory 
figure_name <- "/US_ECig_Flavor_Ban_Map.png"
ggsave(paste0(output_directory, figure_name), plot = figure, width = 14, height = 8, dpi = 300)

# Confirm results have been written to a file
if (file.exists(paste0(output_directory, figure_name))) 
{
  cat("Figure has been written to", paste0(output_directory, figure_name), "\n")
} else 
{
  cat("Error: Figure could not be written\n")
}


