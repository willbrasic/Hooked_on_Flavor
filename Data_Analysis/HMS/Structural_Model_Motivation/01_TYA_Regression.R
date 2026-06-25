################################################################################
# William Brasic 
# The Unvarersity of Arizona
# wbrasic97@gmail.com
# September 2025
#
# Estimates the effect of teens and young adults in the household
# on the probability of purchasing flavored e-cigs
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

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt_tob <- fread(file_name)

# Variables used across all regressions
vars_all <- c(
  "flavored_ecig",
  "teen_present",
  "young_adult_present",
  "teen_or_young_adult_present",
  "household_size",
  "purchase_month",
  "fips_state_code"
)


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
# Teen effect on purchasing
# flavored ecigs for 
# all household-months
############################# 

# Restrict to complete cases on required variables
dt_complete <- dt[complete.cases(dt[, ..vars_all])]

# Different LPM RHS variables
vars <- list(
  teen_present = list(label = "HH Contains a Teen", table = "teen_regression"),
  young_adult_present = list(label = "HH Contains a Young Adult", table = "youngadult_regression"),
  teen_or_young_adult_present = list(label = "HH Contains a Teen or Young Adult", table = "teen_youngadult_regression")
)

# Loop over each RHS variable
for (var in names(vars)) 
{
  # (1) Baseline: var + month FE
  formula_1 <- as.formula(paste0("flavored_ecig ~ ", var, " + i(purchase_month)"))
  model_1 <- plm(formula_1, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # (2) Add household size
  formula_2 <- as.formula(paste0("flavored_ecig ~ ", var, " + household_size + i(purchase_month)"))
  model_2 <- plm(formula_2, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # (3) Add state FE
  formula_3 <- as.formula(paste0("flavored_ecig ~ ", var, " + household_size + i(fips_state_code) + i(purchase_month)"))
  model_3 <- plm(formula_3, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # Cluster-robust variance-covariance matrix
  V1 <- vcovHC(model_1, type = "HC1", cluster = "group")
  V2 <- vcovHC(model_2, type = "HC1", cluster = "group")
  V3 <- vcovHC(model_3, type = "HC1", cluster = "group")
  
  # Variables to keep in the table
  keep_vars <- c(var, "household_size")
  
  # Get robust SEs for those variables in the table
  robust_1 <- get_robust(model_1, V1, keep_vars)
  robust_2 <- get_robust(model_2, V2, keep_vars)
  robust_3 <- get_robust(model_3, V3, keep_vars)
  
  # FE indicators
  add_lines <- list(
    c("State Fixed Effects", "", "", "$\\checkmark$"),
    c("Month Fixed Effects", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$")
  )
  
  # Generate table
  stargazer(
    model_1, model_2, model_3,
    type = "latex",
    keep = keep_vars,
    se = list(robust_1$se, robust_2$se, robust_3$se),
    p  = list(robust_1$p,  robust_2$p,  robust_3$p),
    dep.var.labels = "Purchased Flavored E-Cigarette",
    column.labels = c("(1)", "(2)", "(3)"),
    covariate.labels = c(vars[[var]]$label, "HH Size"),
    omit.stat = c("f", "ser"),
    add.lines = add_lines,
    keep.stat = c("n", "rsq"),
    no.space = TRUE,
    digits = 3,
    label = vars[[var]]$table
  )
}


#############################
# Teen effect on purchasing
# flavored ecigs for 
# household-months
# containing cig or ecig 
# purchase
############################# 

# Keep only e-cigarette purchase observations
# If using all observations, just make it
# dt_ecig <- dt
dt_cig_ecig <- dt_tob

# Variables used across all regressions
vars_all <- c(
  "flavored_ecig",
  "teen_present",
  "young_adult_present",
  "teen_or_young_adult_present",
  "household_size",
  "purchase_month",
  "fips_state_code"
)

# Restrict to complete cases on required variables
dt_complete <- dt_cig_ecig[complete.cases(dt_cig_ecig[, ..vars_all])]

# Different LPM RHS variables
vars <- list(
  teen_present = list(label = "HH Contains a Teen", table = "teen_regression"),
  young_adult_present = list(label = "HH Contains a Young Adult", table = "youngadult_regression"),
  teen_or_young_adult_present = list(label = "HH Contains a Teen or Young Adult", table = "teen_youngadult_regression")
)

# Loop over each RHS variable
for (var in names(vars)) 
{
  # (1) Baseline: var + month FE
  formula_1 <- as.formula(paste0("flavored_ecig ~ ", var, " + i(purchase_month)"))
  model_1 <- plm(formula_1, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # (2) Add household size
  formula_2 <- as.formula(paste0("flavored_ecig ~ ", var, " + household_size + i(purchase_month)"))
  model_2 <- plm(formula_2, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # (3) Add state FE
  formula_3 <- as.formula(paste0("flavored_ecig ~ ", var, " + household_size + i(fips_state_code) + i(purchase_month)"))
  model_3 <- plm(formula_3, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # Cluster-robust variance-covariance matrix
  V1 <- vcovHC(model_1, type = "HC1", cluster = "group")
  V2 <- vcovHC(model_2, type = "HC1", cluster = "group")
  V3 <- vcovHC(model_3, type = "HC1", cluster = "group")
  
  # Variables to keep in the table
  keep_vars <- c(var, "household_size")
  
  # Get robust SEs for those variables in the table
  robust_1 <- get_robust(model_1, V1, keep_vars)
  robust_2 <- get_robust(model_2, V2, keep_vars)
  robust_3 <- get_robust(model_3, V3, keep_vars)
  
  # FE indicators
  add_lines <- list(
    c("State Fixed Effects", "", "", "$\\checkmark$"),
    c("Month Fixed Effects", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$")
  )
  
  # Generate table
  stargazer(
    model_1, model_2, model_3,
    type = "latex",
    keep = keep_vars,
    se = list(robust_1$se, robust_2$se, robust_3$se),
    p  = list(robust_1$p,  robust_2$p,  robust_3$p),
    dep.var.labels = "Purchased Flavored E-Cigarette",
    column.labels = c("(1)", "(2)", "(3)"),
    covariate.labels = c(vars[[var]]$label, "HH Size"),
    omit.stat = c("f", "ser"),
    add.lines = add_lines,
    keep.stat = c("n", "rsq"),
    no.space = TRUE,
    digits = 3,
    label = vars[[var]]$table
  )
}


#############################
# Teen effect on purchasing
# flavored ecigs for 
# household-months
# containing ecig purchases
############################# 

# Keep only e-cigarette purchase observations
# If using all observations, just make it
# dt_ecig <- dt
dt_ecig <- dt_tob[ecig == 1 | cig_ecig == 1]

# Variables used across all regressions
vars_all <- c(
  "flavored_ecig",
  "teen_present",
  "young_adult_present",
  "teen_or_young_adult_present",
  "household_size",
  "purchase_month",
  "fips_state_code"
)

# Restrict to complete cases on required variables
dt_complete <- dt_ecig[complete.cases(dt_ecig[, ..vars_all])]

# Different LPM RHS variables
vars <- list(
  teen_present = list(label = "HH Contains a Teen", table = "teen_regression"),
  young_adult_present = list(label = "HH Contains a Young Adult", table = "youngadult_regression"),
  teen_or_young_adult_present = list(label = "HH Contains a Teen or Young Adult", table = "teen_youngadult_regression")
)

# Loop over each RHS variable
for (var in names(vars)) 
{
  # (1) Baseline: var + month FE
  formula_1 <- as.formula(paste0("flavored_ecig ~ ", var, " + i(purchase_month)"))
  model_1 <- plm(formula_1, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # (2) Add household size
  formula_2 <- as.formula(paste0("flavored_ecig ~ ", var, " + household_size + i(purchase_month)"))
  model_2 <- plm(formula_2, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # (3) Add state FE
  formula_3 <- as.formula(paste0("flavored_ecig ~ ", var, " + household_size + i(fips_state_code) + i(purchase_month)"))
  model_3 <- plm(formula_3, index = c("household_code", "purchase_month"), model = "pooling", data = dt_complete)
  
  # Cluster-robust variance-covariance matrix
  V1 <- vcovHC(model_1, type = "HC1", cluster = "group")
  V2 <- vcovHC(model_2, type = "HC1", cluster = "group")
  V3 <- vcovHC(model_3, type = "HC1", cluster = "group")
  
  # Variables to keep in the table
  keep_vars <- c(var, "household_size")
  
  # Get robust SEs for those variables in the table
  robust_1 <- get_robust(model_1, V1, keep_vars)
  robust_2 <- get_robust(model_2, V2, keep_vars)
  robust_3 <- get_robust(model_3, V3, keep_vars)
  
  # FE indicators
  add_lines <- list(
    c("State Fixed Effects", "", "", "$\\checkmark$"),
    c("Month Fixed Effects", "$\\checkmark$", "$\\checkmark$", "$\\checkmark$")
  )
  
  # Generate table
  stargazer(
    model_1, model_2, model_3,
    type = "latex",
    keep = keep_vars,
    se = list(robust_1$se, robust_2$se, robust_3$se),
    p  = list(robust_1$p,  robust_2$p,  robust_3$p),
    dep.var.labels = "Purchased Flavored E-Cigarette",
    column.labels = c("(1)", "(2)", "(3)"),
    covariate.labels = c(vars[[var]]$label, "HH Size"),
    omit.stat = c("f", "ser"),
    add.lines = add_lines,
    keep.stat = c("n", "rsq"),
    no.space = TRUE,
    digits = 3,
    label = vars[[var]]$table
  )
}










