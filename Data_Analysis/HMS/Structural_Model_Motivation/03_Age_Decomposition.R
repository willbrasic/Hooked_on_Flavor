################################################################################
# William Brasic
# The Unvarersity of Arizona
# wbrasic97@gmail.com
# September 2025
#
# Estimates the effect of household age composition on cigarette and e-cigarette
# consumption using a two-part model (extensive and intensive margins). Mean
# household age is constructed from head and member age columns.
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
pacman::p_load(data.table, plm, sandwich, lmtest, fixest, broom, stargazer, ggplot2, scales, gridExtra)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Image output directory
output_directory <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/",
                              "4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures/")

# Load in full panel data
file_name <- paste0("./all_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt <- fread(file_name)


#############################
# Household Sum of Ages
#############################

# Compute sum of ages across all household members (heads + non-head members).
# Head age columns use 0 to indicate absence; Member_X_Age uses NA.
member_age_cols <- paste0("Member_", 1:7, "_Age")
dt[, sum_age := {
  m_head  <- fifelse(male_head_age   > 0, as.numeric(male_head_age),   NA_real_)
  f_head  <- fifelse(female_head_age > 0, as.numeric(female_head_age), NA_real_)
  age_mat <- cbind(m_head, f_head, as.matrix(.SD))
  rowSums(age_mat, na.rm = TRUE)
}, .SDcols = member_age_cols]

summary(dt[, sum_age])
sd(dt[, sum_age], na.rm = TRUE)


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
# Age effect on cigarette
# purchase probability
# (extensive margin, all HH)
#############################

# Required variables for cigarette purchase probability regressions
vars_cig_ext <- c("cig", "sum_age", "household_size", "purchase_month", "fips_state_code")

# Restrict to complete cases on required variables
dt_cig_ext <- dt[complete.cases(dt[, ..vars_cig_ext])]

# (1) Baseline: sum_age + month FE
model_cig_ext_1 <- plm(
  as.formula("cig ~ sum_age + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_cig_ext
)

# (2) Add household size
model_cig_ext_2 <- plm(
  as.formula("cig ~ sum_age + household_size + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_cig_ext
)

# (3) Add state FE
model_cig_ext_3 <- plm(
  as.formula("cig ~ sum_age + household_size + i(fips_state_code) + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_cig_ext
)

# Cluster-robust variance-covariance matrix
V_cig_ext_1 <- vcovHC(model_cig_ext_1, type = "HC1", cluster = "group")
V_cig_ext_2 <- vcovHC(model_cig_ext_2, type = "HC1", cluster = "group")
V_cig_ext_3 <- vcovHC(model_cig_ext_3, type = "HC1", cluster = "group")

# Variables to keep in the table
keep_vars_ext <- c("sum_age", "household_size")

# Get robust SEs for those variables in the table
robust_cig_ext_1 <- get_robust(model_cig_ext_1, V_cig_ext_1, keep_vars_ext)
robust_cig_ext_2 <- get_robust(model_cig_ext_2, V_cig_ext_2, keep_vars_ext)
robust_cig_ext_3 <- get_robust(model_cig_ext_3, V_cig_ext_3, keep_vars_ext)

# FE indicators
add_lines_ext <- list(
  c("State Fixed Effects", "", "", "$\\checkmark$"),
  c("Month Fixed Effects", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$")
)

# Generate table
stargazer(
  model_cig_ext_1, model_cig_ext_2, model_cig_ext_3,
  type             = "latex",
  keep             = keep_vars_ext,
  se               = list(robust_cig_ext_1$se, robust_cig_ext_2$se, robust_cig_ext_3$se),
  p                = list(robust_cig_ext_1$p,  robust_cig_ext_2$p,  robust_cig_ext_3$p),
  dep.var.labels   = "Purchased Cigarettes (Excl. Bundles)",
  column.labels    = c("(1)", "(2)", "(3)"),
  covariate.labels = c("HH Sum of Ages (Std.)", "HH Size"),
  omit.stat        = c("f", "ser"),
  add.lines        = add_lines_ext,
  keep.stat        = c("n", "rsq"),
  no.space         = TRUE,
  digits           = 3,
  label            = "cig_ext_regression"
)


#############################
# Age effect on e-cigarette
# purchase probability
# (extensive margin, all HH)
#############################

# Required variables for e-cigarette purchase probability regressions
vars_ecig_ext <- c("ecig", "sum_age", "household_size", "purchase_month", "fips_state_code")

# Restrict to complete cases on required variables
dt_ecig_ext <- dt[complete.cases(dt[, ..vars_ecig_ext])]

# (1) Baseline: sum_age + month FE
model_ecig_ext_1 <- plm(
  as.formula("ecig ~ sum_age + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_ecig_ext
)

# (2) Add household size
model_ecig_ext_2 <- plm(
  as.formula("ecig ~ sum_age + household_size + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_ecig_ext
)

# (3) Add state FE
model_ecig_ext_3 <- plm(
  as.formula("ecig ~ sum_age + household_size + i(fips_state_code) + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_ecig_ext
)

# Cluster-robust variance-covariance matrix
V_ecig_ext_1 <- vcovHC(model_ecig_ext_1, type = "HC1", cluster = "group")
V_ecig_ext_2 <- vcovHC(model_ecig_ext_2, type = "HC1", cluster = "group")
V_ecig_ext_3 <- vcovHC(model_ecig_ext_3, type = "HC1", cluster = "group")

# Variables to keep in the table
keep_vars_ext <- c("sum_age", "household_size")

# Get robust SEs for those variables in the table
robust_ecig_ext_1 <- get_robust(model_ecig_ext_1, V_ecig_ext_1, keep_vars_ext)
robust_ecig_ext_2 <- get_robust(model_ecig_ext_2, V_ecig_ext_2, keep_vars_ext)
robust_ecig_ext_3 <- get_robust(model_ecig_ext_3, V_ecig_ext_3, keep_vars_ext)

# FE indicators
add_lines_ext <- list(
  c("State Fixed Effects", "", "", "$\\checkmark$"),
  c("Month Fixed Effects", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$")
)

# Generate table
stargazer(
  model_ecig_ext_1, model_ecig_ext_2, model_ecig_ext_3,
  type             = "latex",
  keep             = keep_vars_ext,
  se               = list(robust_ecig_ext_1$se, robust_ecig_ext_2$se, robust_ecig_ext_3$se),
  p                = list(robust_ecig_ext_1$p,  robust_ecig_ext_2$p,  robust_ecig_ext_3$p),
  dep.var.labels   = "Purchased E-Cigarettes (Excl. Bundles)",
  column.labels    = c("(1)", "(2)", "(3)"),
  covariate.labels = c("HH Sum of Ages (Std.)", "HH Size"),
  omit.stat        = c("f", "ser"),
  add.lines        = add_lines_ext,
  keep.stat        = c("n", "rsq"),
  no.space         = TRUE,
  digits           = 3,
  label            = "ecig_ext_regression"
)


# Extensive margin base rate (full sample)
mean(dt$ecig, na.rm = TRUE)

# Conditional on the filtered sample (ecig/bundle purchasers)
mean(dt_ecig_ext$ecig, na.rm = TRUE)


#############################
# Age effect on cigarette
# consumption quantity
# (packs purchased per month)
#############################

# Required variables for cigarette quantity regressions
vars_cig_qty <- c("total_packs", "sum_age", "household_size", "purchase_month", "fips_state_code")

# All household-months (unconditional: captures extensive + intensive margin)
dt_cig_all <- dt[complete.cases(dt[, ..vars_cig_qty])]

# Cigarette purchaser months only (intensive margin: conditional on cig purchase)
dt_cig_purch <- dt[total_packs > 0 & complete.cases(dt[, ..vars_cig_qty])]

# (1) All HH: sum_age + hh_size + month FE
model_cig_1 <- plm(
  as.formula("total_packs ~ sum_age + household_size + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_cig_all
)

# (2) All HH: + state FE
model_cig_2 <- plm(
  as.formula("total_packs ~ sum_age + household_size + i(fips_state_code) + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_cig_all
)

# (3) Cig purchasers only: sum_age + hh_size + month FE
model_cig_3 <- plm(
  as.formula("total_packs ~ sum_age + household_size + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_cig_purch
)

# (4) Cig purchasers only: + state FE
model_cig_4 <- plm(
  as.formula("total_packs ~ sum_age + household_size + i(fips_state_code) + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_cig_purch
)

# Cluster-robust variance-covariance matrix
V_cig_1 <- vcovHC(model_cig_1, type = "HC1", cluster = "group")
V_cig_2 <- vcovHC(model_cig_2, type = "HC1", cluster = "group")
V_cig_3 <- vcovHC(model_cig_3, type = "HC1", cluster = "group")
V_cig_4 <- vcovHC(model_cig_4, type = "HC1", cluster = "group")

# Variables to keep in the table
keep_vars_qty <- c("sum_age", "household_size")

# Get robust SEs for those variables in the table
robust_cig_1 <- get_robust(model_cig_1, V_cig_1, keep_vars_qty)
robust_cig_2 <- get_robust(model_cig_2, V_cig_2, keep_vars_qty)
robust_cig_3 <- get_robust(model_cig_3, V_cig_3, keep_vars_qty)
robust_cig_4 <- get_robust(model_cig_4, V_cig_4, keep_vars_qty)

# FE indicators
add_lines_cig <- list(
  c("Sample",              "All HH",        "All HH",        "Cig. Purchasers", "Cig. Purchasers"),
  c("State Fixed Effects", "",              "$\\checkmark$", "",                "$\\checkmark$"),
  c("Month Fixed Effects", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$",   "$\\checkmark$")
)

# Generate table
stargazer(
  model_cig_1, model_cig_2, model_cig_3, model_cig_4,
  type              = "latex",
  keep              = keep_vars_qty,
  se                = list(robust_cig_1$se, robust_cig_2$se, robust_cig_3$se, robust_cig_4$se),
  p                 = list(robust_cig_1$p,  robust_cig_2$p,  robust_cig_3$p,  robust_cig_4$p),
  dep.var.labels    = "Cigarette Packs Purchased",
  column.labels     = c("(1)", "(2)", "(3)", "(4)"),
  covariate.labels  = c("HH Sum of Ages", "HH Size"),
  omit.stat         = c("f", "ser"),
  add.lines         = add_lines_cig,
  keep.stat         = c("n", "rsq"),
  no.space          = TRUE,
  digits            = 3,
  label             = "cig_qty_regression"
)


#############################
# Age effect on e-cigarette
# consumption quantity
# (mL purchased per month)
#############################

# Required variables for e-cigarette quantity regressions
vars_ecig_qty <- c("total_mL", "sum_age", "household_size", "purchase_month", "fips_state_code")

# All household-months (unconditional: captures extensive + intensive margin)
dt_ecig_all <- dt[complete.cases(dt[, ..vars_ecig_qty])]

# E-cigarette purchaser months only (intensive margin: conditional on ecig purchase)
dt_ecig_purch <- dt[total_mL > 0 & complete.cases(dt[, ..vars_ecig_qty])]

# (1) All HH: sum_age + hh_size + month FE
model_ecig_1 <- plm(
  as.formula("total_mL ~ sum_age + household_size + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_ecig_all
)

# (2) All HH: + state FE
model_ecig_2 <- plm(
  as.formula("total_mL ~ sum_age + household_size + i(fips_state_code) + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_ecig_all
)

# (3) Ecig purchasers only: sum_age + hh_size + month FE
model_ecig_3 <- plm(
  as.formula("total_mL ~ sum_age + household_size + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_ecig_purch
)

# (4) Ecig purchasers only: + state FE
model_ecig_4 <- plm(
  as.formula("total_mL ~ sum_age + household_size + i(fips_state_code) + i(purchase_month)"),
  index = c("household_code", "purchase_month"), model = "pooling", data = dt_ecig_purch
)

# Cluster-robust variance-covariance matrix
V_ecig_1 <- vcovHC(model_ecig_1, type = "HC1", cluster = "group")
V_ecig_2 <- vcovHC(model_ecig_2, type = "HC1", cluster = "group")
V_ecig_3 <- vcovHC(model_ecig_3, type = "HC1", cluster = "group")
V_ecig_4 <- vcovHC(model_ecig_4, type = "HC1", cluster = "group")

# Get robust SEs for those variables in the table
robust_ecig_1 <- get_robust(model_ecig_1, V_ecig_1, keep_vars_qty)
robust_ecig_2 <- get_robust(model_ecig_2, V_ecig_2, keep_vars_qty)
robust_ecig_3 <- get_robust(model_ecig_3, V_ecig_3, keep_vars_qty)
robust_ecig_4 <- get_robust(model_ecig_4, V_ecig_4, keep_vars_qty)

# FE indicators
add_lines_ecig <- list(
  c("Sample",              "All HH",        "All HH",         "Ecig Purchasers", "Ecig Purchasers"),
  c("State Fixed Effects", "",              "$\\checkmark$",  "",                "$\\checkmark$"),
  c("Month Fixed Effects", "$\\checkmark$", "$\\checkmark$",  "$\\checkmark$",   "$\\checkmark$")
)

# Generate table
stargazer(
  model_ecig_1, model_ecig_2, model_ecig_3, model_ecig_4,
  type              = "latex",
  keep              = keep_vars_qty,
  se                = list(robust_ecig_1$se, robust_ecig_2$se, robust_ecig_3$se, robust_ecig_4$se),
  p                 = list(robust_ecig_1$p,  robust_ecig_2$p,  robust_ecig_3$p,  robust_ecig_4$p),
  dep.var.labels    = "E-Cigarette Volume Purchased (mL)",
  column.labels     = c("(1)", "(2)", "(3)", "(4)"),
  covariate.labels  = c("HH Sum of Ages", "HH Size"),
  omit.stat         = c("f", "ser"),
  add.lines         = add_lines_ecig,
  keep.stat         = c("n", "rsq"),
  no.space          = TRUE,
  digits            = 3,
  label             = "ecig_qty_regression"
)
