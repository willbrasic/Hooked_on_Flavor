################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# Creates the summary stats table for the paper.
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
                "4th_Year_Paper_Data/HMS/2021-Onward/")
setwd(wd)

# Load packages
pacman::p_load(data.table)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Load in all household data
file_name <- paste0("./panelists_information_2021-Onward.rds")
dt_households <- readRDS(file_name)

# Load in tobacco household data
file_name <- paste0("./tobacco_panelists_information_2021-Onward.rds")
dt_tobacco_households <- readRDS(file_name)

# Load in non-tobacco household data
file_name <- paste0("./non-tobacco_panelists_information_2021-Onward.rds")
dt_nontobacco_households <- readRDS(file_name)

# Load in cig only household data (those household years in which a household purchased only cigs in that year)
file_name <- paste0("./cig_only_panelists_information_2021-Onward.rds")
dt_cig_households <- readRDS(file_name)

# Load in ecig only household data (those household years in which a household purchased only ecigs in that year)
file_name <- paste0("./ecig_only_panelists_information_2021-Onward.rds")
dt_ecig_households <- readRDS(file_name)

# Load in cig and ecig household data (those household years in which a household purchased cigs and ecigs in that year)
file_name <- paste0("./cig_ecig_both_panelists_information_2021-Onward.rds")
dt_both_households <- readRDS(file_name)

# Load in PUMS census data
file_name <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/", 
                       "4th_Year_Paper_Data/Census/census_households_persons_2021-Onward.csv")
dt_census <- fread(file_name)

# Top code census income at 100K b/c that's how the Nielsen data is
dt_census[HINCP > 100000, HINCP := 100000]


#############################
# Analysis on all households
############################# 

# Number of unique households
dt_tobacco_households[, uniqueN(household_code)]
dt_nontobacco_households[, uniqueN(household_code)]
dt_census[, uniqueN(SERIALNO)]

# Mean of household income
dt_tobacco_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, mean(household_income, na.rm = TRUE)]
dt_nontobacco_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, mean(household_income, na.rm = TRUE)]
dt_census[, .(HINCP = HINCP[1]), by = SERIALNO][, mean(HINCP, na.rm = TRUE)]

# Standard deviation of household income
dt_tobacco_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, sd(household_income, na.rm = TRUE)]
dt_nontobacco_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, sd(household_income, na.rm = TRUE)]
dt_census[, .(HINCP = HINCP[1]), by = SERIALNO][, sd(HINCP, na.rm = TRUE)]

# Mean head-of-household age
dt_tobacco_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, mean(head_age, na.rm = TRUE)]
dt_nontobacco_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, mean(head_age, na.rm = TRUE)]
dt_census[, .(HHLDRAGEP = HHLDRAGEP[1]), by = SERIALNO][, mean(HHLDRAGEP, na.rm = TRUE)]

# Standard deviation of head-of-household age
dt_tobacco_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, sd(head_age, na.rm = TRUE)]
dt_nontobacco_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, sd(head_age, na.rm = TRUE)]
dt_census[, .(HHLDRAGEP = HHLDRAGEP[1]), by = SERIALNO][, sd(HHLDRAGEP, na.rm = TRUE)]

# Fraction of head-of-household with at least a college degree
dt_tobacco_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, mean(least_one_head_college, na.rm = TRUE)]
dt_nontobacco_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, mean(least_one_head_college, na.rm = TRUE)]
dt_census[, .(head_college = any(SPORDER == 1 & SCHL >= 21, na.rm = TRUE)), by = SERIALNO][, mean(head_college, na.rm = TRUE)]

# Standard deviation of head-of-household with at least a college degree
dt_tobacco_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, sd(least_one_head_college, na.rm = TRUE)]
dt_nontobacco_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, sd(least_one_head_college, na.rm = TRUE)]
dt_census[, .(head_college = any(SPORDER == 1 & SCHL >= 21, na.rm = TRUE)), by = SERIALNO][, sd(head_college, na.rm = TRUE)]

# Fraction of households that are white
dt_tobacco_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, mean(white_race, na.rm = TRUE)]
dt_nontobacco_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, mean(white_race, na.rm = TRUE)]
dt_census[, .(white_race = any(SPORDER == 1 & HHLDRRAC1P == 1, na.rm = TRUE)), by = SERIALNO][, mean(white_race, na.rm = TRUE)]

# Standard deviation of households that are white
dt_tobacco_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, sd(white_race, na.rm = TRUE)]
dt_nontobacco_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, sd(white_race, na.rm = TRUE)]
dt_census[, .(white_race = any(SPORDER == 1 & HHLDRRAC1P == 1, na.rm = TRUE)), by = SERIALNO][, sd(white_race, na.rm = TRUE)]

# Fraction of married households
dt_tobacco_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, mean(marital_status, na.rm = TRUE)]
dt_nontobacco_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, mean(marital_status, na.rm = TRUE)]
dt_census[, .(marital_status = any(SPORDER == 1 & HHT == 1, na.rm = TRUE)), by = SERIALNO][, mean(marital_status, na.rm = TRUE)]

# Standard deviation of married households
dt_tobacco_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, sd(marital_status, na.rm = TRUE)]
dt_nontobacco_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, sd(marital_status, na.rm = TRUE)]
dt_census[, .(marital_status = any(SPORDER == 1 & HHT == 1, na.rm = TRUE)), by = SERIALNO][, sd(marital_status, na.rm = TRUE)]

# Fraction of households with children (aged 0 - 12, inclusive)
dt_tobacco_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, mean(child_present, na.rm = TRUE)]
dt_nontobacco_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, mean(child_present, na.rm = TRUE)]
dt_census[, .(child_present = any(AGEP <= 12, na.rm = TRUE)), by = SERIALNO][, mean(child_present, na.rm = TRUE)]

# Standard deviation of households with children (aged 0 - 12, inclusive)
dt_tobacco_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, sd(child_present, na.rm = TRUE)]
dt_nontobacco_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, sd(child_present, na.rm = TRUE)]
dt_census[, .(child_present = any(AGEP <= 12, na.rm = TRUE)), by = SERIALNO][, sd(child_present, na.rm = TRUE)]

# Fraction of households with teen (aged 13 - 18, inclusive)
dt_tobacco_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, mean(teen_present, na.rm = TRUE)]
dt_nontobacco_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, mean(teen_present, na.rm = TRUE)]
dt_census[, .(teen_present = any(AGEP >= 13 & AGEP <= 18, na.rm = TRUE)), by = SERIALNO][, mean(teen_present, na.rm = TRUE)]

# Standard deviation of households with teen (aged 13 - 18, inclusive)
dt_tobacco_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, sd(teen_present, na.rm = TRUE)]
dt_nontobacco_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, sd(teen_present, na.rm = TRUE)]
dt_census[, .(teen_present = any(AGEP >= 13 & AGEP <= 18, na.rm = TRUE)), by = SERIALNO][, sd(teen_present, na.rm = TRUE)]

# Fraction of households with young adult (aged 19 - 25, inclusive)
dt_tobacco_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, mean(young_adult_present, na.rm = TRUE)]
dt_nontobacco_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, mean(young_adult_present, na.rm = TRUE)]
dt_census[, .(young_adult_present = any(AGEP >= 19 & AGEP <= 25, na.rm = TRUE)), by = SERIALNO][, mean(young_adult_present, na.rm = TRUE)]

# Standard deviation of households with young adult (aged 19 - 25, inclusive)
dt_tobacco_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, sd(young_adult_present, na.rm = TRUE)]
dt_nontobacco_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, sd(young_adult_present, na.rm = TRUE)]
dt_census[, .(young_adult_present = any(AGEP >= 19 & AGEP <= 25, na.rm = TRUE)), by = SERIALNO][, sd(young_adult_present, na.rm = TRUE)]


#############################
# Analysis on all households
# by purchased category
############################# 

# Number of unique households
dt_cig_households[, uniqueN(household_code)]
dt_ecig_households[, uniqueN(household_code)]
dt_both_households[, uniqueN(household_code)]

# Mean of household income
dt_cig_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, mean(household_income, na.rm = TRUE)]
dt_ecig_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, mean(household_income, na.rm = TRUE)]
dt_both_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, mean(household_income, na.rm = TRUE)]

# Standard deviation of household income
dt_cig_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, sd(household_income, na.rm = TRUE)]
dt_ecig_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, sd(household_income, na.rm = TRUE)]
dt_both_households[, .(household_income = mean(household_income, na.rm = TRUE)), by = household_code][, sd(household_income, na.rm = TRUE)]

# Mean head-of-household age
dt_cig_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, mean(head_age, na.rm = TRUE)]
dt_ecig_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, mean(head_age, na.rm = TRUE)]
dt_both_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, mean(head_age, na.rm = TRUE)]

# Standard deviation of head-of-household age
dt_cig_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, sd(head_age, na.rm = TRUE)]
dt_ecig_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, sd(head_age, na.rm = TRUE)]
dt_both_households[, .(head_age = mean(c(male_head_age, female_head_age), na.rm = TRUE)), by = household_code][, sd(head_age, na.rm = TRUE)]

# Fraction of head-of-household with at least a college degree
dt_cig_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, mean(least_one_head_college, na.rm = TRUE)]
dt_ecig_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, mean(least_one_head_college, na.rm = TRUE)]
dt_both_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, mean(least_one_head_college, na.rm = TRUE)]

# Standard deviation of head-of-household with at least a college degree
dt_cig_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, sd(least_one_head_college, na.rm = TRUE)]
dt_ecig_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, sd(least_one_head_college, na.rm = TRUE)]
dt_both_households[, .(least_one_head_college = mean(as.numeric(as.character(least_one_head_college)), na.rm = TRUE)), by = household_code][, sd(least_one_head_college, na.rm = TRUE)]

# Fraction of households that are white
dt_cig_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, mean(white_race, na.rm = TRUE)]
dt_ecig_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, mean(white_race, na.rm = TRUE)]
dt_both_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, mean(white_race, na.rm = TRUE)]

# Standard deviation of households that are white
dt_cig_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, sd(white_race, na.rm = TRUE)]
dt_ecig_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, sd(white_race, na.rm = TRUE)]
dt_both_households[, .(white_race = mean(as.numeric(as.character(white_race)), na.rm = TRUE)), by = household_code][, sd(white_race, na.rm = TRUE)]

# Fraction of married households
dt_cig_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, mean(marital_status, na.rm = TRUE)]
dt_ecig_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, mean(marital_status, na.rm = TRUE)]
dt_both_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, mean(marital_status, na.rm = TRUE)]

# Standard deviation of married households
dt_cig_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, sd(marital_status, na.rm = TRUE)]
dt_ecig_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, sd(marital_status, na.rm = TRUE)]
dt_both_households[, .(marital_status = mean(as.numeric(as.character(marital_status)), na.rm = TRUE)), by = household_code][, sd(marital_status, na.rm = TRUE)]

# Fraction of households with children (aged 0 - 12, inclusive)
dt_cig_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, mean(child_present, na.rm = TRUE)]
dt_ecig_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, mean(child_present, na.rm = TRUE)]
dt_both_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, mean(child_present, na.rm = TRUE)]

# Standard deviation of households with children (aged 0 - 12, inclusive)
dt_cig_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, sd(child_present, na.rm = TRUE)]
dt_ecig_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, sd(child_present, na.rm = TRUE)]
dt_both_households[, .(child_present = mean(child_present, na.rm = TRUE)), by = household_code][, sd(child_present, na.rm = TRUE)]

# Fraction of households with teen (aged 13 - 18, inclusive)
dt_cig_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, mean(teen_present, na.rm = TRUE)]
dt_ecig_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, mean(teen_present, na.rm = TRUE)]
dt_both_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, mean(teen_present, na.rm = TRUE)]

# Standard deviation of households with teen (aged 13 - 18, inclusive)
dt_cig_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, sd(teen_present, na.rm = TRUE)]
dt_ecig_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, sd(teen_present, na.rm = TRUE)]
dt_both_households[, .(teen_present = mean(teen_present, na.rm = TRUE)), by = household_code][, sd(teen_present, na.rm = TRUE)]

# Fraction of households with young adult (aged 19 - 25, inclusive)
dt_cig_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, mean(young_adult_present, na.rm = TRUE)]
dt_ecig_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, mean(young_adult_present, na.rm = TRUE)]
dt_both_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, mean(young_adult_present, na.rm = TRUE)]

# Standard deviation of households with young adult (aged 19 - 25, inclusive)
dt_cig_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, sd(young_adult_present, na.rm = TRUE)]
dt_ecig_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, sd(young_adult_present, na.rm = TRUE)]
dt_both_households[, .(young_adult_present = mean(young_adult_present, na.rm = TRUE)), by = household_code][, sd(young_adult_present, na.rm = TRUE)]


#############################
# Random household facts
############################# 

# Share of households found in all years
dt_tobacco_households[, .N, by=.(household_code, purchase_year)][, .N, by = household_code][, mean(N == uniqueN(dt_tobacco_households[, purchase_year]))]

# Number of unique households by year
dt_households[, uniqueN(household_code), by = .(purchase_year)]
dt_tobacco_households[, uniqueN(household_code), by = .(purchase_year)]

# Number of households where teen or young adult presence changes over time
dt_households[
  , .(n_unique = uniqueN(teen_or_young_adult_present)),
  by = household_code
][n_unique > 1, .N]

















