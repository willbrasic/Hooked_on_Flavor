################################################################################
# William Brasic
# The Unvarersity of Arizona
# wbrasic97@gmail.com
# May 2026
#
# Tests for stockpiling behavior in cigarette and e-cigarette purchases.
# The key test exploits the opposing predictions of addiction vs. stockpiling
# for the transition to zero-purchase months following a high-purchase month:
#
#   1[c_it = 0] = a + b*c_{i,t-1} + HH FE + Month FE + e_it
#
# Addiction predicts b < 0: high past consumption raises addiction stock,
# making zero purchases less likely this month.
# Stockpiling predicts b > 0: high past purchases drew down future demand,
# making zero purchases more likely this month.
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
pacman::p_load(data.table, plm, sandwich, lmtest, fixest, broom, stargazer)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt <- fread(file_name)


#############################
# Sequential Month Index
# and Consumption Lags
#############################

# Convert Date purchase_month to sequential index for gap detection
dt[, month_seq := year(as.IDate(purchase_month)) * 12 + month(as.IDate(purchase_month))]

# Sort by household and month for shift operations
setorder(dt, household_code, month_seq)

# Gap indicator: TRUE if previous observation is the consecutive month
dt[, lag_consec := (month_seq - shift(month_seq, 1)) == 1, by = household_code]

# Lagged consumption within household (NA at panel breaks)
dt[, c_cig_lag  := fifelse(lag_consec == TRUE, shift(total_packs, 1), NA_real_), by = household_code]
dt[, c_ecig_lag := fifelse(lag_consec == TRUE, shift(total_mL,    1), NA_real_), by = household_code]

# Zero-purchase binary outcomes
dt[, zero_cig  := as.integer(total_packs == 0)]
dt[, zero_ecig := as.integer(total_mL    == 0)]


#############################
# Functions
#############################

# Function to extract aligned coef / se / p for kept variables
get_robust <- function(model, V, keep_vars)
{
  b  <- coef(model)
  se <- sqrt(diag(V))
  tt <- b / se
  p  <- 2 * pnorm(-abs(tt))  # large-sample normal p-values

  list(
    se = se[names(b) %in% keep_vars],
    p  = p[names(b) %in% keep_vars]
  )
}


#############################
# Zero-purchase transition
# test for cigarettes
#############################

# Required variables for cigarette transition regressions
vars_cig <- c("zero_cig", "c_cig_lag", "household_code", "purchase_month")

# Restrict to complete cases on required variables
dt_cig <- dt[complete.cases(dt[, ..vars_cig])]

# (1) Lagged consumption + month FE
model_cig_1 <- plm(
  as.formula("zero_cig ~ c_cig_lag + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_cig
)

# (2) Add household FE (two-way within estimator)
model_cig_2 <- plm(
  as.formula("zero_cig ~ c_cig_lag"),
  index = c("household_code", "purchase_month"), model = "within", effect = "twoways", data = dt_cig
)

# Cluster-robust variance-covariance matrix
V_cig_1 <- vcovHC(model_cig_1, type = "HC1", cluster = "group")
V_cig_2 <- vcovHC(model_cig_2, type = "HC1", cluster = "group")

# Variables to keep in the table
keep_vars_cig <- c("c_cig_lag")

# Get robust SEs for those variables in the table
robust_cig_1 <- get_robust(model_cig_1, V_cig_1, keep_vars_cig)
robust_cig_2 <- get_robust(model_cig_2, V_cig_2, keep_vars_cig)

# FE indicators
add_lines_cig <- list(
  c("Household Fixed Effects", "",              "$\\checkmark$"),
  c("Month Fixed Effects",     "$\\checkmark$", "$\\checkmark$")
)

# Generate table
stargazer(
  model_cig_1, model_cig_2,
  type             = "latex",
  keep             = keep_vars_cig,
  se               = list(robust_cig_1$se, robust_cig_2$se),
  p                = list(robust_cig_1$p,  robust_cig_2$p),
  dep.var.labels   = "Zero Cigarette Purchase Month",
  column.labels    = c("(1)", "(2)"),
  covariate.labels = c("Lagged Cig. Packs"),
  omit.stat        = c("f", "ser"),
  add.lines        = add_lines_cig,
  keep.stat        = c("n", "rsq"),
  no.space         = TRUE,
  digits           = 6,
  label            = "cig_stockpiling_test"
)


#############################
# Zero-purchase transition
# test for e-cigarettes
#############################

# Required variables for e-cigarette transition regressions
vars_ecig <- c("zero_ecig", "c_ecig_lag", "household_code", "purchase_month")

# Restrict to complete cases on required variables
dt_ecig <- dt[complete.cases(dt[, ..vars_ecig])]

# (1) Lagged consumption + month FE
model_ecig_1 <- plm(
  as.formula("zero_ecig ~ c_ecig_lag + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_ecig
)

# (2) Add household FE (two-way within estimator)
model_ecig_2 <- plm(
  as.formula("zero_ecig ~ c_ecig_lag"),
  index = c("household_code", "purchase_month"), model = "within", effect = "twoways", data = dt_ecig
)

# Cluster-robust variance-covariance matrix
V_ecig_1 <- vcovHC(model_ecig_1, type = "HC1", cluster = "group")
V_ecig_2 <- vcovHC(model_ecig_2, type = "HC1", cluster = "group")

# Variables to keep in the table
keep_vars_ecig <- c("c_ecig_lag")

# Get robust SEs for those variables in the table
robust_ecig_1 <- get_robust(model_ecig_1, V_ecig_1, keep_vars_ecig)
robust_ecig_2 <- get_robust(model_ecig_2, V_ecig_2, keep_vars_ecig)

# FE indicators
add_lines_ecig <- list(
  c("Household Fixed Effects", "",              "$\\checkmark$"),
  c("Month Fixed Effects",     "$\\checkmark$", "$\\checkmark$")
)

# Generate table
stargazer(
  model_ecig_1, model_ecig_2,
  type             = "latex",
  keep             = keep_vars_ecig,
  se               = list(robust_ecig_1$se, robust_ecig_2$se),
  p                = list(robust_ecig_1$p,  robust_ecig_2$p),
  dep.var.labels   = "Zero E-Cigarette Purchase Month",
  column.labels    = c("(1)", "(2)"),
  covariate.labels = c("Lagged E-Cig. mL"),
  omit.stat        = c("f", "ser"),
  add.lines        = add_lines_ecig,
  keep.stat        = c("n", "rsq"),
  no.space         = TRUE,
  digits           = 6,
  label            = "ecig_stockpiling_test"
)

