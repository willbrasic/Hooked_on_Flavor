################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# September 2026
#
# Tests whether within-household consumption trends over the sample. The
# initial conditions algorithm finds each household's starting addiction
# stock by fixed-point iteration, feeding the household's observed purchases
# back into the starting value until it settles, which assumes the household's
# pre-sample purchasing looked like its in-sample purchasing. A within-household
# time trend violates that assumption: a household ramping up carries a lower
# pre-sample stock than the fixed point implies, and one winding down carries
# a higher one.
#
# Estimates q_it = alpha_i + beta_t + epsilon_it with household and month
# fixed effects, so beta_t traces the average within-household movement in
# quantity net of who happens to be in the panel. Also estimates a single
# linear slope in place of the month effects as a compact test.
#
# Cigarettes use cigarette purchase months (packs) and e-cigarettes use
# e-cigarette purchase months (mL), which tests the intensive margin. A second
# sample includes all months of households that ever purchase the product,
# with zeros, which folds in purchase frequency since the stock depends on
# both margins. The extensive margin is then isolated by regressing the
# purchase indicator itself on the month index, which is where initiation
# would appear as a rising probability of buying. Everything is repeated for
# TYA households, defined as households with a teen or young adult present at
# any point in the panel so that each household's full sequence is kept.
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
pacman::p_load(data.table, fixest)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Figure output directory
output_directory <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/",
                              "4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures/")

# Load in full panel data
file_name <- paste0("./tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.csv")
dt <- fread(file_name)

# Month index (1 = January 2021) for the linear trend regressions
dt[, month_index := as.integer(year(purchase_month) - 2021) * 12L + as.integer(month(purchase_month))]

# Cigarette purchase months include bundles containing cigarettes
dt[, cig_purchase := as.integer(cig == 1 | cig_ecig == 1)]

# E-cigarette purchase months include bundles containing e-cigarettes
dt[, ecig_purchase := as.integer(ecig == 1 | cig_ecig == 1)]


#############################
# Functions
#############################

# Estimates the two-way fixed effects regression and returns the month
# effects relative to January 2021 with cluster-robust confidence intervals
get_month_effects <- function(dt_input, quantity_col, sample_label)
{
  # Month fixed effects with January 2021 as the reference month
  formula_fe <- as.formula(paste0(quantity_col, " ~ i(month_index, ref = 1) | household_code"))
  model_fe <- feols(formula_fe, data = dt_input, cluster = ~household_code)

  # Pull the month effects into a data table
  dt_effects <- as.data.table(coeftable(model_fe), keep.rownames = "term")
  setnames(dt_effects, c("term", "estimate", "se", "t_stat", "p_value"))
  dt_effects[, month_index := as.integer(gsub("month_index::", "", term))]

  # Add the reference month at zero
  dt_effects <- rbind(
    data.table(term = "month_index::1", estimate = 0, se = 0, t_stat = NA_real_, p_value = NA_real_, month_index = 1L),
    dt_effects
  )

  # Confidence bands and labels
  dt_effects[, lo := estimate - 1.96 * se]
  dt_effects[, hi := estimate + 1.96 * se]
  dt_effects[, purchase_month := as.IDate(sprintf("%d-%02d-01", 2021L + (month_index - 1L) %/% 12L, (month_index - 1L) %% 12L + 1L))]
  dt_effects[, sample := sample_label]

  setorder(dt_effects, month_index)
  dt_effects[, .(sample, month_index, purchase_month, estimate, se, lo, hi)]
}

# Estimates a single linear slope on the month index with household fixed
# effects and reports it in units per month and as a share of mean quantity
get_linear_trend <- function(dt_input, quantity_col, sample_label)
{
  formula_trend <- as.formula(paste0(quantity_col, " ~ month_index | household_code"))
  model_trend <- feols(formula_trend, data = dt_input, cluster = ~household_code)

  slope <- coef(model_trend)["month_index"]
  se <- se(model_trend)["month_index"]
  mean_q <- dt_input[, mean(get(quantity_col))]

  data.table(
    sample = sample_label,
    households = dt_input[, uniqueN(household_code)],
    household_months = nrow(dt_input),
    mean_quantity = mean_q,
    slope_per_month = slope,
    se = se,
    t_stat = slope / se,
    slope_pct_of_mean_per_month = 100 * slope / mean_q,
    slope_pct_of_mean_over_36_months = 100 * slope * 35 / mean_q
  )
}


#############################
# Cigarette time trend
#############################

# Intensive margin: household-months with a cigarette purchase
dt_cig_purchase <- dt[cig_purchase == 1]

# Both margins: all household-months for households that ever purchase cigarettes
dt_cig_all <- dt[household_code %in% dt_cig_purchase[, unique(household_code)]]

# Month effects for both samples
dt_cig_effects <- rbind(
  get_month_effects(dt_cig_purchase, "total_packs", "purchase_months"),
  get_month_effects(dt_cig_all, "total_packs", "all_months")
)

# Linear trend for both samples
dt_cig_trend <- rbind(
  get_linear_trend(dt_cig_purchase, "total_packs", "purchase_months"),
  get_linear_trend(dt_cig_all, "total_packs", "all_months")
)

cat("\n=== Cigarettes: Linear Trend in Packs (Household FE) ===\n")
print(dt_cig_trend)

cat("\n=== Cigarettes: Month Effects Relative to January 2021 (Purchase Months) ===\n")
print(dt_cig_effects[sample == "purchase_months", .(purchase_month, estimate, se, lo, hi)])


#############################
# E-cigarette time trend
#############################

# Intensive margin: household-months with an e-cigarette purchase
dt_ecig_purchase <- dt[ecig_purchase == 1]

# Both margins: all household-months for households that ever purchase e-cigarettes
dt_ecig_all <- dt[household_code %in% dt_ecig_purchase[, unique(household_code)]]

# Month effects for both samples
dt_ecig_effects <- rbind(
  get_month_effects(dt_ecig_purchase, "total_mL", "purchase_months"),
  get_month_effects(dt_ecig_all, "total_mL", "all_months")
)

# Linear trend for both samples
dt_ecig_trend <- rbind(
  get_linear_trend(dt_ecig_purchase, "total_mL", "purchase_months"),
  get_linear_trend(dt_ecig_all, "total_mL", "all_months")
)

cat("\n=== E-Cigarettes: Linear Trend in mL (Household FE) ===\n")
print(dt_ecig_trend)

cat("\n=== E-Cigarettes: Month Effects Relative to January 2021 (Purchase Months) ===\n")
print(dt_ecig_effects[sample == "purchase_months", .(purchase_month, estimate, se, lo, hi)])


#############################
# TYA households
#############################

# Households with a teen or young adult present at any point in the panel
dt_tya <- dt[teen_or_young_adult_ever == 1]

# Cigarettes: intensive margin and both margins
dt_tya_cig_purchase <- dt_tya[cig_purchase == 1]
dt_tya_cig_all <- dt_tya[household_code %in% dt_tya_cig_purchase[, unique(household_code)]]

dt_tya_cig_effects <- rbind(
  get_month_effects(dt_tya_cig_purchase, "total_packs", "purchase_months"),
  get_month_effects(dt_tya_cig_all, "total_packs", "all_months")
)

dt_tya_cig_trend <- rbind(
  get_linear_trend(dt_tya_cig_purchase, "total_packs", "purchase_months"),
  get_linear_trend(dt_tya_cig_all, "total_packs", "all_months")
)

cat("
=== TYA Households, Cigarettes: Linear Trend in Packs (Household FE) ===
")
print(dt_tya_cig_trend)

# E-cigarettes: intensive margin and both margins
dt_tya_ecig_purchase <- dt_tya[ecig_purchase == 1]
dt_tya_ecig_all <- dt_tya[household_code %in% dt_tya_ecig_purchase[, unique(household_code)]]

dt_tya_ecig_effects <- rbind(
  get_month_effects(dt_tya_ecig_purchase, "total_mL", "purchase_months"),
  get_month_effects(dt_tya_ecig_all, "total_mL", "all_months")
)

dt_tya_ecig_trend <- rbind(
  get_linear_trend(dt_tya_ecig_purchase, "total_mL", "purchase_months"),
  get_linear_trend(dt_tya_ecig_all, "total_mL", "all_months")
)

cat("
=== TYA Households, E-Cigarettes: Linear Trend in mL (Household FE) ===
")
print(dt_tya_ecig_trend)


#############################
# Extensive margin time trend
#############################

# Probability of any purchase in a month, for households that ever purchase
# the product, so the slope is the within-household trend in purchase frequency

# All households
dt_cig_ext_effects <- get_month_effects(dt_cig_all, "cig_purchase", "all_months")
dt_cig_ext_trend <- get_linear_trend(dt_cig_all, "cig_purchase", "all_months")
dt_ecig_ext_effects <- get_month_effects(dt_ecig_all, "ecig_purchase", "all_months")
dt_ecig_ext_trend <- get_linear_trend(dt_ecig_all, "ecig_purchase", "all_months")

cat("
=== Extensive Margin: Linear Trend in Purchase Probability (Household FE) ===
")
print(rbind(dt_cig_ext_trend[, product := "cigarettes"], dt_ecig_ext_trend[, product := "e-cigarettes"]))

# TYA households
dt_tya_cig_ext_effects <- get_month_effects(dt_tya_cig_all, "cig_purchase", "all_months")
dt_tya_cig_ext_trend <- get_linear_trend(dt_tya_cig_all, "cig_purchase", "all_months")
dt_tya_ecig_ext_effects <- get_month_effects(dt_tya_ecig_all, "ecig_purchase", "all_months")
dt_tya_ecig_ext_trend <- get_linear_trend(dt_tya_ecig_all, "ecig_purchase", "all_months")

cat("
=== TYA Households, Extensive Margin: Linear Trend in Purchase Probability (Household FE) ===
")
print(rbind(dt_tya_cig_ext_trend[, product := "cigarettes"], dt_tya_ecig_ext_trend[, product := "e-cigarettes"]))


#############################
# Save month effects for
# plotting
#############################

# Tag each result with its household group and product
dt_cig_effects[, group := "all_households"]
dt_ecig_effects[, group := "all_households"]
dt_tya_cig_effects[, group := "tya_households"]
dt_tya_ecig_effects[, group := "tya_households"]
dt_cig_trend[, `:=`(group = "all_households", product = "cigarettes")]
dt_ecig_trend[, `:=`(group = "all_households", product = "e-cigarettes")]
dt_tya_cig_trend[, `:=`(group = "tya_households", product = "cigarettes")]
dt_tya_ecig_trend[, `:=`(group = "tya_households", product = "e-cigarettes")]
dt_cig_ext_effects[, `:=`(group = "all_households", margin = "extensive")]
dt_ecig_ext_effects[, `:=`(group = "all_households", margin = "extensive")]
dt_tya_cig_ext_effects[, `:=`(group = "tya_households", margin = "extensive")]
dt_tya_ecig_ext_effects[, `:=`(group = "tya_households", margin = "extensive")]
dt_cig_ext_trend[, `:=`(group = "all_households", margin = "extensive")]
dt_ecig_ext_trend[, `:=`(group = "all_households", margin = "extensive")]
dt_tya_cig_ext_trend[, `:=`(group = "tya_households", margin = "extensive")]
dt_tya_ecig_ext_trend[, `:=`(group = "tya_households", margin = "extensive")]
for (dt_q in list(dt_cig_effects, dt_ecig_effects, dt_tya_cig_effects, dt_tya_ecig_effects,
                  dt_cig_trend, dt_ecig_trend, dt_tya_cig_trend, dt_tya_ecig_trend))
{
  dt_q[, margin := "quantity"]
}

# Save to the figures directory for pgfplots
fwrite(rbind(dt_cig_effects, dt_tya_cig_effects, dt_cig_ext_effects, dt_tya_cig_ext_effects),
       file.path(output_directory, "10_Time_Trend_Month_Effects_Cig.txt"))
fwrite(rbind(dt_ecig_effects, dt_tya_ecig_effects, dt_ecig_ext_effects, dt_tya_ecig_ext_effects),
       file.path(output_directory, "10_Time_Trend_Month_Effects_ECig.txt"))
fwrite(rbind(dt_cig_trend, dt_ecig_trend, dt_tya_cig_trend, dt_tya_ecig_trend,
             dt_cig_ext_trend, dt_ecig_ext_trend, dt_tya_cig_ext_trend, dt_tya_ecig_ext_trend),
       file.path(output_directory, "10_Time_Trend_Linear_Slopes.txt"))
