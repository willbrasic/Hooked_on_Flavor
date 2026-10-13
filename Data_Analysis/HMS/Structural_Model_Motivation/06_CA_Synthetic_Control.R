################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# Synthetic control for California cigarette and unflavored e-cigarette prices
# around the December 2022 CA flavor ban (statute's legal effective date, Dec.
# 21, 2022). Other ban states are excluded from the donor pool.
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
pacman::p_load(data.table, tidysynth, ggplot2)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# CA flavor ban date and all states with flavor bans (FIPS codes)
ban_date             <- as.IDate("2022-12-01")
list_fips_ban_states <- c(6, 25, 34, 36, 44)

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

# Dec 2022 = treatment time index
ban_time_idx <- as.integer(format(as.Date("2022-12-01"), "%Y")) * 12 +
                as.integer(format(as.Date("2022-12-01"), "%m")) - origin_month + 1

# Use three non-collinear predictors: early pre-treatment, late pre-treatment, and
# the month immediately before the ban. Using multiple highly correlated price
# averages leads to near-singular predictor matrices, so I avoid that here.

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
    generate_predictor(time_window = 12:23, price_late  = mean(median_price, na.rm = TRUE)) |>
    generate_weights(optimization_window = 1:23) |>
    generate_control()

sc_cig |> plot_trends()      + labs(title = "Cigarette Prices - CA vs Synthetic Control",    y = "Median Real Price ($/pack)")
sc_cig |> plot_differences() + labs(title = "Cigarette Prices - CA minus Synthetic Control", y = "Median Real Price ($/pack)")
sc_cig |> plot_weights()     + labs(title = "Cigarette Prices - Donor State Weights")
sc_cig |> plot_placebos()    + labs(title = "Cigarette Prices - Placebo Tests")


#############################
# Synthetic Control: CA Purchase Rates
#############################

# Quasi-experimental motivation for the structural model. The price synthetic
# controls above ask whether the ban moved prices; this section asks what the
# ban did to household behavior. The outcome for each state-month is the share
# of panel households purchasing (i) a flavored e-cigarette, (ii) any
# e-cigarette, (iii) cigarettes, and (iv) any tobacco product. If the ban binds
# only partially, shifts some households toward cigarettes rather than out of
# nicotine, and does so gradually, that is the pattern the dynamic model with
# addiction and habit stocks is built to rationalize and that a static or
# reduced-form estimate cannot extrapolate to a federal ban, a tax, or
# enforcement.
#
# Power caveat: only 16 CA households buy a flavored e-cigarette in the twelve
# months before the ban and roughly 250-300 CA households are observed per
# month, so the e-cigarette outcomes rest on a handful of purchases per month.
# Cigarette and any-tobacco purchase rates are well measured.

# Household-level purchase indicators for the four outcomes
dt_monthly[, any_tobacco    := 1L - outside_option]
dt_monthly[, unflavored_ecig := as.integer(ecig == 1 & flavored_ecig == 0)]

# State-month purchase rates; exclude other ban states from the donor pool and
# require at least 30 households per state-month so rates are not dominated by
# one or two households
min_hh_per_state_month <- 30

dt_sc_rates <- dt_monthly[
    !fips_state_code %in% list_fips_ban_states[list_fips_ban_states != 6],
    .(
        n_hh               = uniqueN(household_code),
        rate_flavored_ecig = mean(flavored_ecig),
        rate_any_ecig      = mean(ecig),
        rate_cig           = mean(cig),
        rate_any_tobacco   = mean(any_tobacco)
    ),
    keyby = .(fips_state_code, purchase_month)
][n_hh >= min_hh_per_state_month]

# Balanced panel of states across all months (required by Synth)
list_states_balanced_rates <- dt_sc_rates[, .N, keyby = fips_state_code][N == n_months_total, fips_state_code]
dt_sc_rates <- dt_sc_rates[fips_state_code %in% list_states_balanced_rates]

# Numeric time index (months since Jan 2021 = 1), as above
dt_sc_rates[, time_idx := as.integer(format(as.Date(purchase_month), "%Y")) * 12 +
                          as.integer(format(as.Date(purchase_month), "%m")) - origin_month + 1]

cat("\nCA purchase-rate synthetic control: donor pool has",
    length(list_states_balanced_rates) - 1, "states;",
    dt_sc_rates[fips_state_code == 6, paste(range(n_hh), collapse = "-")],
    "CA households per month\n")

# Pre- and post-ban windows (ban month itself counts as post)
list_pre_idx  <- 1:(ban_time_idx - 1)
list_post_idx <- ban_time_idx:n_months_total

# Fit one synthetic control per outcome. Predictors are the early and late
# pre-period means of the outcome, matching the price specification above, so
# that the same donor weighting logic applies to every outcome.
run_rate_synth <- function(outcome_name) {

    dt_in <- dt_sc_rates[, .(fips_state_code, time_idx, y = get(outcome_name))]

    # A donor state whose outcome never varies over the pre-ban window (e.g. no
    # flavored purchases in a small state) makes the placebo weight problem
    # singular, so such states are dropped from this outcome's donor pool
    degenerate <- dt_in[time_idx %in% list_pre_idx, .(sd_y = sd(y)), by = fips_state_code][is.na(sd_y) | sd_y == 0, fips_state_code]
    degenerate <- setdiff(degenerate, 6)
    if (length(degenerate) > 0) cat("  ", outcome_name, ": dropping", length(degenerate), "donor state(s) with a constant pre-ban outcome\n")
    dt_in <- as.data.frame(dt_in[!fips_state_code %in% degenerate])

    # If the weight problem still fails, report it and skip this outcome rather
    # than stopping the script
    tryCatch(
        dt_in |>
            synthetic_control(
                outcome           = y,
                unit              = fips_state_code,
                time              = time_idx,
                i_unit            = 6,
                i_time            = ban_time_idx,
                generate_placebos = TRUE
            ) |>
            generate_predictor(time_window = 1:11,  y_early = mean(y, na.rm = TRUE)) |>
            generate_predictor(time_window = 12:23, y_late  = mean(y, na.rm = TRUE)) |>
            generate_weights(optimization_window = list_pre_idx) |>
            generate_control(),
        error = function(e) {
            cat("  ", outcome_name, ": synthetic control failed and is skipped:", conditionMessage(e), "\n")
            NULL
        }
    )
}

list_outcomes <- c(
    rate_flavored_ecig = "Flavored E-Cig. Purchase Rate",
    rate_any_ecig      = "Any E-Cig. Purchase Rate",
    rate_cig           = "Cigarette Purchase Rate",
    rate_any_tobacco   = "Any Tobacco Purchase Rate"
)

list_sc_rates <- lapply(names(list_outcomes), run_rate_synth)
names(list_sc_rates) <- names(list_outcomes)
list_sc_rates <- Filter(Negate(is.null), list_sc_rates)

# Plots: CA versus synthetic CA, the gap, and the placebo-in-space distribution
for (outcome_name in names(list_sc_rates)) {
    label <- list_outcomes[[outcome_name]]
    print(list_sc_rates[[outcome_name]] |> plot_trends()      + labs(title = paste(label, "- CA vs Synthetic Control"),    y = "Share of Households"))
    print(list_sc_rates[[outcome_name]] |> plot_differences() + labs(title = paste(label, "- CA minus Synthetic Control"), y = "Share of Households"))
    print(list_sc_rates[[outcome_name]] |> plot_placebos()    + labs(title = paste(label, "- Placebo Tests"),              y = "Share of Households"))
}

# Summary table: pre-ban fit, mean post-ban gap (levels and percent of the
# synthetic CA post-ban mean), and the permutation p-value from ranking CA's
# post/pre RMSPE ratio against every placebo state
dt_sc_summary <- rbindlist(lapply(names(list_sc_rates), function(outcome_name) {

    dt_series <- as.data.table(grab_synthetic_control(list_sc_rates[[outcome_name]]))
    setnames(dt_series, c("time_unit", "real_y", "synth_y"), c("time_idx", "ca", "synth_ca"))
    dt_series[, gap := ca - synth_ca]

    dt_sig <- as.data.table(grab_significance(list_sc_rates[[outcome_name]]))[unit_name == 6]

    data.table(
        outcome            = list_outcomes[[outcome_name]],
        pre_rmspe          = dt_series[time_idx %in% list_pre_idx,  sqrt(mean(gap^2))],
        pre_mean_gap       = dt_series[time_idx %in% list_pre_idx,  mean(gap)],
        post_mean_gap      = dt_series[time_idx %in% list_post_idx, mean(gap)],
        post_mean_gap_pct  = dt_series[time_idx %in% list_post_idx, 100 * mean(gap) / mean(synth_ca)],
        post_pre_mspe_ratio = dt_sig$mspe_ratio,
        permutation_p      = dt_sig$fishers_exact_pvalue,
        n_placebos         = nrow(as.data.table(grab_unit_weights(list_sc_rates[[outcome_name]])))
    )
}))

cat("\nCA flavor ban: synthetic control on purchase rates\n")
print(dt_sc_summary, digits = 3)

# Month-by-month series (CA, synthetic CA, gap) for all four outcomes
dt_sc_rates_series <- rbindlist(lapply(names(list_sc_rates), function(outcome_name) {
    dt_series <- as.data.table(grab_synthetic_control(list_sc_rates[[outcome_name]]))
    setnames(dt_series, c("time_unit", "real_y", "synth_y"), c("time_idx", "ca", "synth_ca"))
    dt_series[, `:=`(
        outcome        = outcome_name,
        purchase_month = as.IDate(seq(as.Date("2021-01-01"), by = "month", length.out = n_months_total))[time_idx],
        gap            = ca - synth_ca,
        post_ban       = as.integer(time_idx >= ban_time_idx)
    )]
    dt_series[, .(outcome, time_idx, purchase_month, post_ban, ca, synth_ca, gap)]
}))

# Save to the figures directory for pgfplots
fig_dir <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper",
                     "4th_Year_Paper/4th_Year_Paper_Figures")
fwrite(dt_sc_rates_series, file.path(fig_dir, "06_CA_SC_Purchase_Rates.txt"))
fwrite(dt_sc_summary,      file.path(fig_dir, "06_CA_SC_Purchase_Rates_Summary.txt"))
