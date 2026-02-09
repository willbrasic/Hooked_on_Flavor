################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script generates the table for the category-level transition matrix.
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
options(datatable.print.nrows = 20)

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt <- fread(file_name)


#############################
# Transition matrix
#############################

# Panel columns
id_col   <- "household_code"
time_col <- "purchase_month"

# Create mutually exclusive state indicators
dt_states <- copy(dt)

# Create new mutually exclusive state columns
dt_states[, `:=` (
  outside_option = fifelse(cig == 0 & ecig == 0 & cig_ecig == 0, 1, 0),
  cig = fifelse(cig == 1 & ecig == 0 & cig_ecig == 0, 1, 0),
  original_ecig = fifelse(cig == 0 & ecig == 1 & cig_ecig == 0 & original_ecig == 1 & flavored_ecig == 0, 1, 0),
  flavored_ecig = fifelse(cig == 0 & ecig == 1 & cig_ecig == 0 & original_ecig == 0 & flavored_ecig == 1, 1, 0),
  cig_and_original_ecig = fifelse(cig == 0 & ecig == 0 & cig_ecig == 1 & original_ecig == 1 & flavored_ecig == 0, 1, 0),
  cig_and_flavored_ecig = fifelse(cig == 0 & ecig == 0 & cig_ecig == 1 & original_ecig == 0 & flavored_ecig == 1, 1, 0)
)]

# Define all states (mutually exclusive)
states <- c(
  "outside_option",
  "cig",
  "original_ecig",
  "flavored_ecig",
  "cig_and_original_ecig",
  "cig_and_flavored_ecig"
)

# Keep only needed columns
cols <- c(id_col, time_col, states)
dt_transitions <- dt_states[, ..cols]

# Ensure sorted panel
setkeyv(dt_transitions, c(id_col, time_col))

# Verify mutual exclusivity - each row should have exactly one state = 1
dt_transitions[, n_states := rowSums(.SD), .SDcols = states]
dt_transitions <- dt_transitions[n_states == 1]
dt_transitions[, n_states := NULL]

# Assign state variable
dt_transitions[, state := NA_character_]
for (s in states) dt_transitions[get(s) == 1, state := s]

# Drop any rows where state is still NA
dt_transitions <- dt_transitions[!is.na(state)]

# Lagged state within household
dt_transitions[, lag_state := shift(state), by = id_col]
dt_transitions <- dt_transitions[!is.na(lag_state)]

# Tabulate transitions (lag_state -> state) with full support
trans <- dt_transitions[, .N, by = .(lag_state, state)]
trans <- trans[CJ(lag_state = states, state = states), on = .(lag_state, state)]
trans[is.na(N), N := 0L]

# Counts matrix
mat_counts <- dcast(trans, lag_state ~ state, value.var = "N", fill = 0L)
mat_counts <- mat_counts[match(states, lag_state)]
setcolorder(mat_counts, c("lag_state", states))

# Row-stochastic transition probability matrix
m <- as.matrix(mat_counts[, ..states])
rownames(m) <- mat_counts$lag_state
P <- m / rowSums(m)
P


#############################
# Transition matrix
# for HH with teens
# or young adults
#############################

# Panel columns
id_col   <- "household_code"
time_col <- "purchase_month"

# Keep only households with teen or young adult present
dt_filtered <- dt[teen_or_young_adult_present == 1]

# Create mutually exclusive state indicators
dt_states <- copy(dt_filtered)

# Drop observations where both original and flavored ecig purchased in same month
dt_states <- dt_states[!(original_ecig == 1 & flavored_ecig == 1)]

# Create new mutually exclusive state columns
dt_states[, `:=` (
  outside_option = fifelse(cig == 0 & ecig == 0 & cig_ecig == 0, 1, 0),
  cig = fifelse(cig == 1 & ecig == 0 & cig_ecig == 0, 1, 0),
  original_ecig = fifelse(cig == 0 & ecig == 1 & cig_ecig == 0 & original_ecig == 1 & flavored_ecig == 0, 1, 0),
  flavored_ecig = fifelse(cig == 0 & ecig == 1 & cig_ecig == 0 & original_ecig == 0 & flavored_ecig == 1, 1, 0),
  cig_and_original_ecig = fifelse(cig == 0 & ecig == 0 & cig_ecig == 1 & original_ecig == 1 & flavored_ecig == 0, 1, 0),
  cig_and_flavored_ecig = fifelse(cig == 0 & ecig == 0 & cig_ecig == 1 & original_ecig == 0 & flavored_ecig == 1, 1, 0)
)]

# Define all states (mutually exclusive)
states <- c(
  "outside_option",
  "cig",
  "original_ecig",
  "flavored_ecig",
  "cig_and_original_ecig",
  "cig_and_flavored_ecig"
)

# Keep only needed columns
cols <- c(id_col, time_col, states)
dt_transitions <- dt_states[, ..cols]

# Ensure sorted panel
setkeyv(dt_transitions, c(id_col, time_col))

# Verify mutual exclusivity - each row should have exactly one state = 1
dt_transitions[, n_states := rowSums(.SD), .SDcols = states]
dt_transitions <- dt_transitions[n_states == 1]
dt_transitions[, n_states := NULL]

# Assign state variable
dt_transitions[, state := NA_character_]
for (s in states) dt_transitions[get(s) == 1, state := s]

# Drop any rows where state is still NA
dt_transitions <- dt_transitions[!is.na(state)]

# Lagged state within household
dt_transitions[, lag_state := shift(state), by = id_col]
dt_transitions <- dt_transitions[!is.na(lag_state)]

# Tabulate transitions (lag_state -> state) with full support
trans <- dt_transitions[, .N, by = .(lag_state, state)]
trans <- trans[CJ(lag_state = states, state = states), on = .(lag_state, state)]
trans[is.na(N), N := 0L]

# Counts matrix
mat_counts <- dcast(trans, lag_state ~ state, value.var = "N", fill = 0L)
mat_counts <- mat_counts[match(states, lag_state)]
setcolorder(mat_counts, c("lag_state", states))

# Row-stochastic transition probability matrix
m <- as.matrix(mat_counts[, ..states])
rownames(m) <- mat_counts$lag_state
P_young <- m / rowSums(m)
P_young








