################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# August 2026
#
# Tests whether flavored e-cigarette purchasing shows persistence beyond what
# overall nicotine consumption alone would predict. Regresses a household's
# current-month flavored e-cigarette purchase indicator on its own prior-month
# flavored purchase indicator, controlling for prior-month total nicotine
# absorbed (a proxy for the household's overall nicotine dependence level).
# A significant coefficient on lagged flavored purchase, controlling for
# nicotine consumption, indicates flavor-specific state dependence distinct
# from nicotine addiction, motivating the separate flavored habit stock in
# the structural model.
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

# Load in full tobacco purchasers panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt_tob <- fread(file_name)


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
# Build lagged flavor purchase
# and lagged nicotine controls
#############################

dt <- copy(dt_tob)
setkeyv(dt, c("household_code", "purchase_month"))

# Lag the flavored purchase indicator and total nicotine absorbed within household
dt[, lag_flavored_ecig := shift(flavored_ecig), by = household_code]
# Prior-month flavored e-liquid volume in mL (zero in months with no flavored purchase)
dt[is.na(total_flavored_mL), total_flavored_mL := 0]
dt[, lag_flavored_mL := shift(total_flavored_mL), by = household_code]


#############################
# Flavor persistence,
# controlling for nicotine
# consumption
#############################

# Variables used across all regressions
vars_all <- c(
  "flavored_ecig",
  "lag_flavored_ecig",
  "lag_flavored_mL",
  "purchase_month",
  "fips_state_code"
)

# Restrict to complete cases on required variables
dt_complete <- dt[complete.cases(dt[, ..vars_all])]

# Condition on e-cigarette purchase occasions
dt_complete <- dt_complete[ecig == 1 | cig_ecig == 1]

# (1) Baseline: lagged flavored purchase + state FE + month FE
formula_1 <- as.formula("flavored_ecig ~ lag_flavored_ecig + i(household_code) + i(purchase_month)")
model_1 <- plm(formula_1, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)

# (2) Add lagged flavored mL (overall e-cigarette consumption control)
formula_2 <- as.formula("flavored_ecig ~ lag_flavored_ecig + lag_flavored_mL + i(household_code) + i(purchase_month)")
model_2 <- plm(formula_2, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)

# Cluster-robust variance-covariance matrices
V1 <- vcovHC(model_1, type = "HC1", cluster = "group")
V2 <- vcovHC(model_2, type = "HC1", cluster = "group")

# Variables to keep in the table
keep_vars <- c("lag_flavored_ecig", "lag_flavored_mL")

# Get robust SEs for those variables in the table
robust_1 <- get_robust(model_1, V1, keep_vars)
robust_2 <- get_robust(model_2, V2, keep_vars)

# FE indicators
add_lines <- list(
  c("Household Fixed Effects", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$"),
  c("Month Fixed Effects", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$")
)

# Generate table
stargazer(
  model_1, model_2,
  type = "latex",
  keep = keep_vars,
  se = list(robust_1$se, robust_2$se),
  p  = list(robust_1$p,  robust_2$p),
  dep.var.labels = "Purchased Flavored E-Cigarette",
  column.labels = c("(1)", "(2)"),
  covariate.labels = c("Purchased Flavored E-Cig., Prior Month", "Flavored E-Liquid (mL), Prior Month"),
  omit.stat = c("f", "ser"),
  add.lines = add_lines,
  keep.stat = c("n", "rsq"),
  no.space = TRUE,
  digits = 5,
  label = "flavor_persistence_regression"
)
