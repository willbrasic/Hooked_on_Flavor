################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script generates category-level month-to-month transition matrices for
# tobacco product choices. Transition matrices are computed for (i) all
# households that purchase tobacco-related products and (ii) the subset of
# households with a teen or young adult (ages 13-25) present.
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

# Load in full panel data
dt <- fread(paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv"))

# Panel columns
id_col   <- "household_code"
time_col <- "purchase_month"


#############################
# Define mutually exclusive
# product category states
#############################

# Six mutually exclusive states:
#   1. outside_option:        no tobacco purchased
#   2. cig:                   cigarettes only
#   3. original_ecig:         original (unflavored) e-cig only
#   4. flavored_ecig:         flavored e-cig only
#   5. cig_and_original_ecig: cigarettes + original e-cig bundle
#   6. cig_and_flavored_ecig: cigarettes + flavored e-cig bundle

states <- c(
  "outside_option",
  "cig",
  "original_ecig",
  "flavored_ecig",
  "cig_and_original_ecig",
  "cig_and_flavored_ecig"
)


#############################
# Helper function:
# compute transition matrix
#############################

# Takes a data.table with the raw panel columns and returns a row-stochastic
# transition probability matrix. Rows are lagged states, columns are current
# states. The function creates mutually exclusive state indicators, assigns
# each observation to a single state, computes the lagged state within each
# household, and tabulates transitions.
compute_transition_matrix <- function(dt_input, states, id_col, time_col)
{
  # Copy to avoid modifying the input
  dt_work <- copy(dt_input)

  # Create mutually exclusive state indicators
  dt_work[, `:=` (
    outside_option        = fifelse(cig == 0 & ecig == 0 & cig_ecig == 0, 1, 0),
    cig                   = fifelse(cig == 1 & ecig == 0 & cig_ecig == 0, 1, 0),
    original_ecig         = fifelse(cig == 0 & ecig == 1 & cig_ecig == 0 &
                                      original_ecig == 1 & flavored_ecig == 0, 1, 0),
    flavored_ecig         = fifelse(cig == 0 & ecig == 1 & cig_ecig == 0 &
                                      original_ecig == 0 & flavored_ecig == 1, 1, 0),
    cig_and_original_ecig = fifelse(cig == 0 & ecig == 0 & cig_ecig == 1 &
                                      original_ecig == 1 & flavored_ecig == 0, 1, 0),
    cig_and_flavored_ecig = fifelse(cig == 0 & ecig == 0 & cig_ecig == 1 &
                                      original_ecig == 0 & flavored_ecig == 1, 1, 0)
  )]

  # Keep only panel identifiers and state columns
  cols <- c(id_col, time_col, states)
  dt_work <- dt_work[, ..cols]

  # Sort by household and time
  setkeyv(dt_work, c(id_col, time_col))

  # Verify mutual exclusivity: each row should have exactly one state = 1
  dt_work[, n_states := rowSums(.SD), .SDcols = states]
  dt_work <- dt_work[n_states == 1]
  dt_work[, n_states := NULL]

  # Assign a single state string per observation
  dt_work[, state := NA_character_]
  for (s in states)
  {
    dt_work[get(s) == 1, state := s]
  }
  dt_work <- dt_work[!is.na(state)]

  # Compute lagged state within each household
  dt_work[, lag_state := shift(state), by = id_col]
  dt_work <- dt_work[!is.na(lag_state)]

  # Tabulate transitions (lag_state -> state) with full support over all
  # state pairs, filling unobserved transitions with zero
  dt_trans <- dt_work[, .(N = .N), by = .(lag_state, state)]
  dt_trans <- dt_trans[CJ(lag_state = states, state = states),
                       on = .(lag_state, state)]
  dt_trans[is.na(N), N := 0]

  # Pivot to a counts matrix (rows = lagged states, columns = current states)
  dt_counts <- dcast(dt_trans, lag_state ~ state, value.var = "N", fill = 0)
  dt_counts <- dt_counts[match(states, lag_state)]
  setcolorder(dt_counts, c("lag_state", states))

  # Compute row-stochastic transition probability matrix
  mat <- as.matrix(dt_counts[, ..states])
  rownames(mat) <- dt_counts$lag_state
  P <- mat / rowSums(mat)

  return(P)
}


#############################
# Transition matrix:
# all households
#############################

# Compute transition matrix
P <- compute_transition_matrix(dt, states, id_col, time_col)
P


#############################
# Transition matrix:
# households with teen
# or young adult present
#############################

# Restrict to household-months where a teen or young adult (ages 13-25) is present
dt_tya <- dt[teen_or_young_adult_present == 1]

# Drop observations where both original and flavored ecig purchased in same
# month, since these cannot be assigned to a single mutually exclusive state
dt_tya <- dt_tya[!(original_ecig == 1 & flavored_ecig == 1)]

# Compute transition matrix
P_young <- compute_transition_matrix(dt_tya, states, id_col, time_col)
P_young
