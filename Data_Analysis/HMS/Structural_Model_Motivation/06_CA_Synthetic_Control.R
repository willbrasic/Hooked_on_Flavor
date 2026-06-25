################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# Synthetic control for California cigarette and unflavored e-cigarette prices
# around the November 2022 CA flavor ban. Other ban states are excluded from
# the donor pool.
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
pacman::p_load(data.table, tidysynth)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# CA flavor ban date and all states with flavor bans (FIPS codes)
ban_date             <- as.IDate("2022-11-01")
list_fips_ban_states <- c(6, 25, 34, 36, 44, 49)

# Load monthly panel
file_name  <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt_monthly <- fread(file_name)


#############################
# Synthetic Control: CA Flavor Ban
#############################

# State-month median prices; exclude other ban states from donor pool.
# Require n_obs >= 5 per state-month to avoid noisy state-month cells.
dt_sc_cig <- dt_monthly[
    cig == 1 & !fips_state_code %in% list_fips_ban_states[list_fips_ban_states != 6],
    .(median_price = median(real_per_pack_price_paid, na.rm = TRUE), n_obs = .N),
    keyby = .(fips_state_code, purchase_month)
][n_obs >= 5]

dt_sc_ecig <- dt_monthly[
    ecig == 1 & flavored_ecig == 0 &
    !fips_state_code %in% list_fips_ban_states[list_fips_ban_states != 6],
    .(median_price = median(real_per_mL_price_paid, na.rm = TRUE), n_obs = .N),
    keyby = .(fips_state_code, purchase_month)
][n_obs >= 2]

# Keep only states with complete coverage across all months (balanced panel required by Synth).
# For ecigs use CA's coverage as the benchmark: only keep donor states present in every
# month that CA is present, and restrict both to that common set of months.
n_months_total <- uniqueN(dt_monthly$purchase_month)

list_states_balanced_cig <- dt_sc_cig[, .N, keyby = fips_state_code][N == n_months_total, fips_state_code]
dt_sc_cig <- dt_sc_cig[fips_state_code %in% list_states_balanced_cig]

list_months_ca_ecig       <- dt_sc_ecig[fips_state_code == 6, purchase_month]
list_states_balanced_ecig <- dt_sc_ecig[
    purchase_month %in% list_months_ca_ecig,
    .N,
    keyby = fips_state_code
][N == length(list_months_ca_ecig), fips_state_code]
dt_sc_ecig <- dt_sc_ecig[
    fips_state_code %in% list_states_balanced_ecig &
    purchase_month  %in% list_months_ca_ecig
]

# Convert purchase_month to numeric time index (months since Jan 2021 = 1)
# tidysynth requires a numeric time variable
origin_month <- as.integer(format(as.Date("2021-01-01"), "%Y")) * 12 +
                as.integer(format(as.Date("2021-01-01"), "%m"))

dt_sc_cig[,  time_idx := as.integer(format(as.Date(purchase_month), "%Y")) * 12 +
                         as.integer(format(as.Date(purchase_month), "%m")) - origin_month + 1]
dt_sc_ecig[, time_idx := as.integer(format(as.Date(purchase_month), "%Y")) * 12 +
                         as.integer(format(as.Date(purchase_month), "%m")) - origin_month + 1]

# Nov 2022 = treatment time index
ban_time_idx <- as.integer(format(as.Date("2022-11-01"), "%Y")) * 12 +
                as.integer(format(as.Date("2022-11-01"), "%m")) - origin_month + 1

# Use three non-collinear predictors: early pre-treatment, late pre-treatment, and
# the month immediately before the ban. Using multiple highly correlated price
# averages leads to near-singular predictor matrices, so we avoid that here.

# Cigarettes
sc_cig <- as.data.frame(dt_sc_cig) |>
    synthetic_control(
        outcome           = median_price,
        unit              = fips_state_code,
        time              = time_idx,
        i_unit            = 6,
        i_time            = ban_time_idx,
        generate_placebos = TRUE
    ) |>
    generate_predictor(time_window = 1:11,  price_early = mean(median_price, na.rm = TRUE)) |>
    generate_predictor(time_window = 12:22, price_late  = mean(median_price, na.rm = TRUE)) |>
    generate_weights(optimization_window = 1:22) |>
    generate_control()

sc_cig |> plot_trends()      + labs(title = "Cigarette Prices - CA vs Synthetic Control",    y = "Median Real Price ($/pack)")
sc_cig |> plot_differences() + labs(title = "Cigarette Prices - CA minus Synthetic Control", y = "Median Real Price ($/pack)")
sc_cig |> plot_weights()     + labs(title = "Cigarette Prices - Donor State Weights")
sc_cig |> plot_placebos()    + labs(title = "Cigarette Prices - Placebo Tests")
