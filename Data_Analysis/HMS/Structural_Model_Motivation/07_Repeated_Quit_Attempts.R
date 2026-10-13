################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# August 2026
#
# Repeated failed quit attempts. Extends 04_Relapse_Analysis.R, which pools
# all quit episodes together, by counting DISTINCT qualifying abstention
# spells per household. Under exponential discounting, a household that finds
# it optimal to stop purchasing should generally stay stopped; repeated
# genuine quit attempts within the same household (quit, relapse, quit
# again) is the classic signature of naive present bias (O'Donoghue and
# Rabin, 1999, 2001): each present self correctly plans to stay quit, a later
# present self re-optimizes and relapses, and because the household is naive
# it tries again rather than learning it cannot sustain abstinence.
#
# A minimum spell-length threshold (K consecutive zero-purchase months) is
# used to separate genuine quit attempts from routine month-to-month gaps in
# ordinarily intermittent purchasing, which even a fully rational household
# would exhibit as prices and addiction stocks fluctuate.
#
# The panel is NOT balanced: 9,675 households average only ~20.8 of 36
# possible months (households enter/exit the panel), confirmed below. Run
# detection therefore checks contiguity: a gap in a household's panel
# presence always starts a new run, regardless of whether the purchase
# indicator value on either side happens to match, so panel attrition is
# never mistaken for a continued (or continued-then-resumed) abstention spell.
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
pacman::p_load(data.table)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 100)

# Minimum consecutive zero-purchase months to count as a genuine quit attempt
# (checked at both 2 and 3 months as a robustness pair)
list_min_spell_length <- c(2, 3)

# Minimum months of panel tenure required for a household to be included, so
# that very short panel spells (mechanically incapable of showing a repeat
# attempt) don't dilute the repeat-attempt share
min_tenure_months <- 12

# Load in full panel data
dt <- fread(paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv"))

# Month index for contiguity checks
dt[, month_idx := as.integer(format(as.Date(purchase_month), "%Y")) * 12 +
                  as.integer(format(as.Date(purchase_month), "%m"))]
last_month_idx <- max(dt$month_idx)

# Panel balance diagnostics (this panel turns out to be unbalanced: households
# enter/exit over the 2021-2023 window)
n_households   <- uniqueN(dt$household_code)
dt_tenure_check <- dt[, .N, by = household_code]
cat(sprintf("Households: %d, Mean months observed: %.1f, Median: %d, Min: %d, Max: %d\n",
            n_households, mean(dt_tenure_check$N), median(dt_tenure_check$N),
            min(dt_tenure_check$N), max(dt_tenure_check$N)))

# Restrict to households with at least min_tenure_months of panel presence
list_hh_sufficient_tenure <- dt_tenure_check[N >= min_tenure_months, household_code]
cat(sprintf("Households with >= %d months of tenure: %d (%.1f%% of all households)\n",
            min_tenure_months, length(list_hh_sufficient_tenure),
            100 * length(list_hh_sufficient_tenure) / n_households))
dt <- dt[household_code %in% list_hh_sufficient_tenure]


#############################
# Function:
# identify qualifying
# abstention spells
#############################

# Returns a data.table of qualifying quit-attempt spells (one row per spell):
# household_code, spell length, and whether the spell is censored (its true
# end, relapse or continued abstention, is not observed, either because
# it runs to the panel's global last month, or because the household exits
# the panel immediately after). A spell only counts if (1) it is a
# contiguous run of zeros of at least min_spell_length consecutive calendar
# months, and (2) it is not the household's very first run: a leading
# zero-run has no observed prior purchase to "quit" from (left-censored, not
# a genuine quit attempt).
compute_quit_spells <- function(dt_input, purchase_col, min_spell_length)
{
  dt_work <- copy(dt_input)
  setorder(dt_work, household_code, month_idx)

  # A new run starts whenever the purchase indicator changes OR there is a
  # gap in the household's panel presence (so attrition/re-entry never gets
  # silently merged into a purchase or abstention run)
  dt_work[, `:=`(
    purchase_lag  = shift(get(purchase_col)),
    month_idx_lag = shift(month_idx)
  ), by = household_code]

  dt_work[, new_run := is.na(purchase_lag) |
                       get(purchase_col) != purchase_lag |
                       month_idx != month_idx_lag + 1]
  dt_work[, run_id := cumsum(new_run), by = household_code]

  dt_runs <- dt_work[, .(
    purchase_val   = get(purchase_col)[1],
    n_months       = .N,
    end_month_idx  = max(month_idx)
  ), by = .(household_code, run_id)]

  # First run per household (excluded if it's a zero-run: left-censored)
  dt_first_run <- dt_work[, .(first_run_id = min(run_id)), by = household_code]
  dt_runs <- merge(dt_runs, dt_first_run, by = "household_code")

  # A household's own last observed month (for panel-exit censoring)
  dt_hh_last_month <- dt_work[, .(hh_last_month_idx = max(month_idx)), by = household_code]
  dt_runs <- merge(dt_runs, dt_hh_last_month, by = "household_code")

  dt_spells <- dt_runs[
    purchase_val == 0 & n_months >= min_spell_length & run_id != first_run_id,
    .(household_code, spell_length = n_months,
      censored = (end_month_idx == last_month_idx) | (end_month_idx == hh_last_month_idx))
  ]

  dt_spells
}


#############################
# Functions:
# per-household spell counts
# and overdispersion check
#############################

# Households "at risk" of a quit attempt: those with at least one purchase
# month in the panel (so a quit attempt is logically possible)
get_at_risk_households <- function(dt_input, purchase_col)
{
  dt_input[, .(any_purchase = max(get(purchase_col))), by = household_code][any_purchase == 1, household_code]
}

# Summarize the distribution of qualifying-spell counts per household, the
# share of ever-attempting households with 2+ distinct attempts, and an
# overdispersion index (Var / Mean) relative to a constant-hazard (Poisson)
# benchmark with the same mean, computed over all at-risk households
# (including the many with zero qualifying spells).
summarize_repeat_attempts <- function(dt_spells, at_risk_households, product_label, min_spell_length)
{
  dt_counts <- dt_spells[, .(n_spells = .N), by = household_code]

  # Bring in at-risk households with zero qualifying spells
  dt_counts <- merge(
    data.table(household_code = at_risk_households),
    dt_counts, by = "household_code", all.x = TRUE
  )
  dt_counts[is.na(n_spells), n_spells := 0]

  n_at_risk    <- nrow(dt_counts)
  n_ever       <- sum(dt_counts$n_spells >= 1)
  n_repeat     <- sum(dt_counts$n_spells >= 2)
  mean_spells  <- mean(dt_counts$n_spells)
  var_spells   <- var(dt_counts$n_spells)
  dispersion   <- var_spells / mean_spells

  cat(sprintf("\n--- %s (min spell length = %d months) ---\n", product_label, min_spell_length))
  cat(sprintf("At-risk households (>=1 purchase in panel): %d\n", n_at_risk))
  cat(sprintf("Households with >=1 qualifying quit attempt: %d (%.1f%% of at-risk)\n",
              n_ever, 100 * n_ever / n_at_risk))
  cat(sprintf("Households with >=2 distinct qualifying quit attempts: %d (%.1f%% of at-risk, %.1f%% of ever-attempters)\n",
              n_repeat, 100 * n_repeat / n_at_risk, 100 * n_repeat / n_ever))
  cat("Distribution of qualifying-spell counts (at-risk households):\n")
  print(dt_counts[, .N, keyby = n_spells])
  cat(sprintf("Mean spells = %.4f, Variance = %.4f, Dispersion index (Var/Mean) = %.3f\n",
              mean_spells, var_spells, dispersion))

  invisible(dt_counts)
}


#############################
# Run for Cigarettes and
# E-Cigarettes, K = 2 and K = 3
#############################

dt[, any_cig  := fifelse(cig == 1 | cig_ecig == 1, 1, 0)]
dt[, any_ecig := fifelse(ecig == 1 | cig_ecig == 1, 1, 0)]

at_risk_cig  <- get_at_risk_households(dt, "any_cig")
at_risk_ecig <- get_at_risk_households(dt, "any_ecig")

for (k in list_min_spell_length)
{
  spells_cig  <- compute_quit_spells(dt, "any_cig",  k)
  spells_ecig <- compute_quit_spells(dt, "any_ecig", k)

  summarize_repeat_attempts(spells_cig,  at_risk_cig,  "Cigarettes",   k)
  summarize_repeat_attempts(spells_ecig, at_risk_ecig, "E-Cigarettes", k)
}
