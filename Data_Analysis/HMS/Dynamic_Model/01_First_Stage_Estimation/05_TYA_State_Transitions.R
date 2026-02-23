################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script computes TYA (teen/young adult) state transition probabilities
# from household member ages. The TYA state is expanded from binary {0, 1} to
# a 4-state variable that captures proximity to transition:
#
#   State 1: No TYA, stable (no child within 2 years of teen age)
#   State 2: No TYA, approaching (oldest child is 11-12)
#   State 3: TYA present, stable (youngest TYA member <= 23)
#   State 4: TYA present, ending soon (youngest TYA member >= 24)
#
# Age definitions (from HMS):
#   Teen: 13-18 (inclusive)
#   Young adult: 19-25 (inclusive)
#   TYA: 13-25, ages out at 26
#
# The transition matrix allows the structural model to capture anticipated
# TYA changes, enabling identification of beta (present bias parameter).
#
# Outputs (to Dynamic_Model/Data/):
#   - TYA_States.csv: household-month TYA state assignments
#   - TYA_Transition_Matrix.csv: monthly transition probabilities
################################################################################


#############################
# Preliminaries
#############################

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# Set working directory (raw panel data)
wd <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper",
                "4th_Year_Paper_Data/HMS/2021-Onward",
                "Tobacco_Panelists_Purchases_2021-Onward")
setwd(wd)

# Output directory
wd_out <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper",
                    "4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")

# Load packages
pacman::p_load(data.table)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 20)


#############################
# Load data
#############################

# Panel columns
id_col   <- "household_code"
time_col <- "purchase_month"

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt <- fread(file_name)

# Member age column names (non-head members only)
age_cols <- paste0("Member_", 1:7, "_Age")

# All age columns (heads + members)
all_age_cols <- c("male_head_age", "female_head_age", age_cols)

# Keep relevant columns
keep_cols <- c(id_col, time_col, "teen_or_young_adult_present", all_age_cols)
dt <- dt[, ..keep_cols]

# Raw CSV is pre-sorted by (household_code, purchase_month) from
# 04_Monthly_Aggregation_2021-Onward.R — no additional sorting needed.


#############################
# Compute youngest TYA
# and oldest child ages
#############################

# All ages as a matrix (heads + members, 9 columns)
mat_ages <- as.matrix(dt[, ..all_age_cols])

# Treat age <= 0 as missing (no member in that slot)
mat_ages[!is.na(mat_ages) & mat_ages <= 0] <- NA

# Youngest member in TYA range (13-25 inclusive)
mat_tya <- mat_ages
mat_tya[is.na(mat_tya) | mat_tya < 13 | mat_tya > 25] <- NA

list_tya <- lapply(seq_len(ncol(mat_tya)), function(k) mat_tya[, k])
youngest_tya <- do.call(pmin, c(list_tya, na.rm = TRUE))
youngest_tya[is.infinite(youngest_tya)] <- NA_real_
dt[, youngest_tya_age := youngest_tya]

# Oldest pre-teen child (0-12 inclusive, non-head members only)
mat_child <- as.matrix(dt[, ..age_cols])
mat_child[is.na(mat_child) | mat_child <= 0 | mat_child > 12] <- NA

list_child <- lapply(seq_len(ncol(mat_child)), function(k) mat_child[, k])
oldest_child <- do.call(pmax, c(list_child, na.rm = TRUE))
oldest_child[is.infinite(oldest_child)] <- NA_real_
dt[, oldest_child_age := oldest_child]


#############################
# Classify TYA states
#############################

# 4-state TYA classification based on proximity to transition:
#   State 1: No TYA, stable — no TYA member AND oldest child <= 10 (or no children)
#   State 2: No TYA, approaching — no TYA member BUT oldest child is 11-12
#   State 3: TYA present, stable — youngest TYA member <= 23
#   State 4: TYA present, ending soon — youngest TYA member >= 24
#
# ~2% of observations have a mismatch between teen_or_young_adult_present
# and the Member_*_Age columns. We use teen_or_young_adult_present as the
# authoritative TYA status and default TYA=1 with missing youngest_tya_age
# to state 3 (stable).

dt[, tya_state := fcase(
  teen_or_young_adult_present == 0 & (is.na(oldest_child_age) | oldest_child_age <= 10), 1L,
  teen_or_young_adult_present == 0 & oldest_child_age >= 11,                              2L,
  teen_or_young_adult_present == 1 & !is.na(youngest_tya_age) & youngest_tya_age <= 23,   3L,
  teen_or_young_adult_present == 1 & !is.na(youngest_tya_age) & youngest_tya_age >= 24,   4L,
  teen_or_young_adult_present == 1 & is.na(youngest_tya_age),                              3L
)]

# State distribution
dt[, .(N = .N, pct = round(.N / nrow(dt) * 100, 2)), keyby = tya_state]


#############################
# Compute monthly transition
# probabilities
#############################

# Lead TYA state (next month's state for the same household)
dt[, tya_state_next := shift(tya_state, n = 1, type = "lead"), by = id_col]

# Drop last observation per household (no lead available)
dt_trans <- dt[!is.na(tya_state_next)]

# TYA states
states <- 1:4

# Tabulate transitions (tya_state -> tya_state_next) with full support
dt_counts <- dt_trans[, .N, keyby = .(from = tya_state, to = tya_state_next)]
dt_counts <- dt_counts[CJ(from = states, to = states), on = .(from, to)]
dt_counts[is.na(N), N := 0L]

# Counts matrix
mat_counts <- dcast(dt_counts, from ~ to, value.var = "N", fill = 0L)
mat_counts <- mat_counts[match(states, from)]
setcolorder(mat_counts, c("from", as.character(states)))

# Row-stochastic transition probability matrix
m <- as.matrix(mat_counts[, -1])
rownames(m) <- mat_counts$from
P <- m / rowSums(m)
P


#############################
# Output
#############################

# File names
file_name_states      <- "TYA_States.csv"
file_name_transitions <- "TYA_Transition_Matrix.csv"

# TYA state assignments (one per household-month observation)
fwrite(dt[, .(tya_state)], file.path(wd_out, file_name_states))

# Transition matrix (from, to, probability) in long format
dt_trans_mat <- CJ(from = states, to = states)
dt_trans_mat[, prob := as.vector(t(P))]
fwrite(dt_trans_mat, file.path(wd_out, file_name_transitions))

# Confirm results have been written
if (file.exists(file.path(wd_out, file_name_states)) & file.exists(file.path(wd_out, file_name_transitions)))
{
  cat("Results have been written to\n", wd_out, "\n")
} else
{
  cat("Error: Results could not be written\n")
}











