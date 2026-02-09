################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# October 2025
#
# This script cleans the panelist info data for non-tobacco panelists. 
################################################################################


#############################
# Preliminaries   
############################# 

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# Set working directory
wd <- "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS"
setwd(wd)

# Load packages
pacman::p_load(data.table)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Vector for years of data we are interested in 
years <- 2021:2023


############################# 
# Get all household info
# for all years
#############################  

# Initialize empty list to store panelist info for each year in years
list_households <- vector("list", length(years))
names(list_households) <- as.character(years)

# Load in panelist data for each year in years and attach it to the list created above
for(year in years)
{
  # Load in data of product hierarchy
  file_name <- paste0("./2021-Onward/", year, "/panelists_", year,  ".tsv")
  dt_panelists <- fread(file_name)
  
  # Attach data table to the list initialized above
  list_households[[as.character(year)]] <- dt_panelists
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}

# Drop columns not needed
cols_to_drop <- c("projection_factor", "projection_factor_magnet",
                  "type_of_residence", "household_composition", "male_head_occupation", 
                  "female_head_occupation", "kitchen_appliances", "tv_items", 
                  "household_internet_connection",  "wic_indicator_current", 
                  "wic_indicator_ever_not_current",
                  "Member_1_Employment",
                  "Member_2_Employment",
                  "Member_3_Employment",
                  "Member_4_Employment",
                  "Member_5_Employment",
                  "Member_6_Employment",
                  "Member_7_Employment")
list_households <- lapply(list_households, function(dt) 
{
  dt[, (cols_to_drop) := NULL]
  
  dt
})

# Rename panel year column
list_households <- lapply(list_households, function(dt) 
{
  setnames(dt, old = "panel_year", new = "purchase_year")
  
  dt
})


############################# 
# Clean household info
############################# 

# Use household income codes from HMS manual to input income midpoint as the household income
income_lower <- c(0, 5000, 8000, 10000, 12000, 15000, 20000, 25000,
                  30000, 35000, 40000, 45000, 50000, 60000, 70000, 100000)
income_upper <- c(5000, 8000, 10000, 12000, 15000, 20000, 25000, 30000,
                  35000, 40000, 45000, 50000, 60000, 70000, 100000, NA_real_)
income_codes <- c(3, 4, 6, 8, 10, 11, 13, 15, 16, 17, 18, 19, 21, 23, 26, 27)
income_values <- setNames(
  ifelse(income_codes == 27, 100000, (income_lower + income_upper) / 2),
  income_codes
)
list_households <- lapply(list_households, function(dt) 
{
  dt[, household_income := income_values[as.character(household_income)] ]
  
  dt
})

# # Get number of households without a male or female head (returns zero for all years as expected)
# lapply(list_households, function(dt)
#   nrow(dt[male_head_age == 0 & female_head_age == 0])
# )

# Adjust male and female head age columns to reflect their actual age using the
# male and female head birth year columns
list_households <- lapply(list_households, function(dt) 
{
  dt[, male_head_age := fifelse(is.na(male_head_birth), NA_integer_, purchase_year - male_head_birth)]
  dt[, female_head_age := fifelse(is.na(female_head_birth), NA_integer_, purchase_year - female_head_birth)]
  
  # Drop birth year columns
  cols_to_drop <- c("male_head_birth", "female_head_birth")
  dt[, (cols_to_drop) := NULL]
  
  dt
})

# Create mean head age column as the mean of the male and female head
# age columns
list_households <- lapply(list_households, function(dt) 
{
  dt[, mean_head_age := apply(.SD, 1, function(x) mean(x, na.rm = TRUE)),
     .SDcols = c("male_head_age", "female_head_age")]
  
  # Reorder columns
  setcolorder(dt, "mean_head_age", after = "female_head_age")
  
  dt
})

# Compute household member ages from member birth year and panel year columns
possible_members <- 7
list_households <- lapply(list_households, function(dt) 
{
  for (k in seq_len(possible_members)) 
  {
    birth_year <- paste0("Member_", k, "_Birth")
    age <- paste0("Member_", k, "_Age")
    dt[, (age) := purchase_year - get(birth_year)]
    dt[, (birth_year) := NULL]
  }
  
  dt
})

# Determine if son or daughter is in the household
list_households <- lapply(list_households, function(dt) 
{
  dt[, son_or_daughter_present := fifelse(
    Member_1_Relationship_Sex %in% c(1, 2) |
      Member_2_Relationship_Sex %in% c(1, 2) |
      Member_3_Relationship_Sex %in% c(1, 2) |
      Member_4_Relationship_Sex %in% c(1, 2) |
      Member_5_Relationship_Sex %in% c(1, 2) |
      Member_6_Relationship_Sex %in% c(1, 2) |
      Member_7_Relationship_Sex %in% c(1, 2),
    1, 0
  )]
  dt[is.na(son_or_daughter_present), son_or_daughter_present := 0]
  
  dt[, paste0("Member_", 1:7, "_Relationship_Sex") := NULL]
  
  setcolorder(dt, "son_or_daughter_present", after = "age_and_presence_of_children")
  
  dt
})

# Create column if child, teen, or young adult lives in the house (including HoHs)
list_households <- lapply(list_households, function(dt) 
{
  age_cols_members <- grep("^Member_\\d+_Age$", names(dt), value = TRUE)
  age_cols_all <- c(age_cols_members, "male_head_age", "female_head_age")
  
  # Child present in household (members + heads)
  dt[, child_count := rowSums(.SD >= 0 & .SD <= 12, na.rm = TRUE), .SDcols = age_cols_all]
  dt[, child_present := fifelse(child_count > 0, 1, 0)]
  dt[is.na(child_present), child_present := 0]
  
  # Teen present in household (members + heads)
  dt[, teen_count := rowSums(.SD >= 13 & .SD <= 18, na.rm = TRUE), .SDcols = age_cols_all]
  dt[, teen_present := fifelse(teen_count > 0, 1, 0)]
  dt[is.na(teen_present), teen_present := 0]
  
  # Young adult present in household (members + heads)
  dt[, young_adult_count := rowSums(.SD >= 19 & .SD <= 25, na.rm = TRUE), .SDcols = age_cols_all]
  dt[, young_adult_present := fifelse(young_adult_count > 0, 1, 0)]
  dt[is.na(young_adult_present), young_adult_present := 0]
  
  # Teen or Young adult present in household
  dt[, teen_or_young_adult_present := fifelse(teen_present == 1 | young_adult_present == 1, 1, 0)]
  dt[is.na(teen_or_young_adult_present), teen_or_young_adult_present := 0]
  dt[, teen_or_young_adult_count := rowSums(.SD >= 13 & .SD <= 25, na.rm = TRUE), .SDcols = age_cols_all]
  
  # Child or Teen or Young adult present in household
  dt[, child_or_teen_or_young_adult_present :=
       fifelse(child_present == 1 | teen_present == 1 | young_adult_present == 1, 1, 0)]
  dt[is.na(child_or_teen_or_young_adult_present), child_or_teen_or_young_adult_present := 0]
  dt[, child_or_teen_or_young_adult_count := rowSums(.SD >= 0 & .SD <= 25, na.rm = TRUE), .SDcols = age_cols_all]
  
  # Reorder columns
  setcolorder(
    dt,
    c("child_present", "child_count",
      "teen_present", "teen_count",
      "young_adult_present", "young_adult_count",
      "teen_or_young_adult_present", "teen_or_young_adult_count",
      "child_or_teen_or_young_adult_present", "child_or_teen_or_young_adult_count"),
    after = "age_and_presence_of_children"
  )
  
  # Drop age and presence of children column
  dt[, "age_and_presence_of_children" := NULL]
  
  dt
})

# Create column for full-time work status for male and female heads 
list_households <- lapply(list_households, function(dt)
{
  dt[, male_head_full_time := fifelse(male_head_employment == 3, 1, 0)]
  dt[, female_head_full_time := fifelse(female_head_employment == 3, 1, 0)]
  dt[, least_one_head_full_time := fifelse(
    (male_head_full_time == 1) | (female_head_full_time == 1), 1, 0
  )]
  
  setcolorder(dt, "male_head_full_time", after = "male_head_employment")
  setcolorder(dt, "female_head_full_time", after = "female_head_employment")
  setcolorder(dt, "least_one_head_full_time", after = "female_head_full_time")
  
  dt[, c("male_head_employment", "female_head_employment") := NULL]
  
  dt
})

# Create column for college graduation status for male and female heads 
list_households <- lapply(list_households, function(dt)
{
  dt[, male_head_college := fifelse(male_head_education == 5 | male_head_education == 6, 1, 0)]
  dt[, female_head_college := fifelse(female_head_education == 5 | female_head_education == 6, 1, 0)]
  dt[, least_one_head_college := fifelse(
    (male_head_college == 1) | (female_head_college == 1), 1, 0
  )]
  
  setcolorder(dt, "male_head_college", after = "male_head_education")
  setcolorder(dt, "female_head_college", after = "female_head_education")
  setcolorder(dt, "least_one_head_college", after = "female_head_college")
  
  dt[, c("male_head_education", "female_head_education") := NULL]
  
  dt
})

# # Check if there are blank values for marital status column
# lapply(list_households, function(dt) unique(dt[, marital_status]))

# Create marital status column to reflect if individual is married or not
list_households <- lapply(list_households, function(dt) 
{
  dt[, marital_status := fifelse(
    is.na(marital_status), 0,
    fifelse(marital_status == 1, 1, 0)
  )]
  
  dt
})

# Create columns for races 
list_households <- lapply(list_households, function(dt) 
{
  dt[, white_race := fifelse(
    is.na(race), 0,
    fifelse(race == 1, 1, 0)
  )]
  
  dt[, other_race := fifelse(
    is.na(race), 1,
    fifelse(race %in% c(2, 3, 4), 1, 0)
  )]
  
  setcolorder(dt, c("white_race", "other_race"), after = "race")
  
  dt[, c("race", "hispanic_origin") := NULL]
  
  dt
})

# Create column for if household is in a state with flavored e-cig bans 
states_with_flavor_bans <- c(6, 25, 34, 36, 44, 49)
list_households <- lapply(list_households, function(dt) 
{
  dt[, state_flavor_ban := fifelse(
    fips_state_code %in% states_with_flavor_bans, 1, 0
  )]
  
  setcolorder(dt, "state_flavor_ban", after = "fips_state_code")
  
  dt
})

# Create list containing only household info for each year
list_households <- lapply(list_households, function(dt) 
{
  cols <- names(dt)
  index  <- match("household_income", cols)
  keep <- c("household_code", "purchase_year", cols[index:length(cols)])
  unique(dt[, ..keep], by = "household_code")
})

# Combine years from `list_households` to form a single data table
dt_households <- rbindlist(list_households)[order(household_code, purchase_year)]

# # Make sure there are no NAs
# colSums(is.na(dt_households))

# Write household information data table to a file
file_name_csv <- "./2021-Onward/panelists_information_2021-Onward.csv"
file_name_rds <- "./2021-Onward/panelists_information_2021-Onward.rds"
fwrite(dt_households, file_name_csv)
saveRDS(dt_households, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds)) 
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else 
{
  cat("Error: File could not be written\n")
}


############################# 
# Filter out all tobacco
# using households
############################# 

# Load in tobacco household data
file_name <- "./2021-Onward/tobacco_panelists_information_2021-Onward.rds"
dt_tobacco_households <- readRDS(file_name)

# Filter out all tobacco using households
dt_nontobacco_households <- dt_households[!household_code %in% dt_tobacco_households[, household_code]]

# Write non-tobacco household information data table to a file
file_name_csv <- "./2021-Onward/non-tobacco_panelists_information_2021-Onward.csv"
file_name_rds <- "./2021-Onward/non-tobacco_panelists_information_2021-Onward.rds"
fwrite(dt_nontobacco_households, file_name_csv)
saveRDS(dt_nontobacco_households, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds)) 
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else 
{
  cat("Error: File could not be written\n")
}















