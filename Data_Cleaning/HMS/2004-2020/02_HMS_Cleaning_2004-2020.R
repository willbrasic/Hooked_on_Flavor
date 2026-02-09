################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script cleans the panelist purchase data created in the 
# HMS_Data_Aggregation_2004-2020.R script within the same folder 
# as this current script. The arbitrary numbers used in some of the lines of 
# code come from the HMS manual located at 
# "../Documentation_2004-2020/Consumer_Panel_Dataset_Manual_2020.pdf." 
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
                "4th_Year_Paper_Data/HMS/2004-2020/Tobacco_Panelists_Purchases_2004-2020/")
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
years <- 2018:2020


#############################
# Load in panelist purchase
# data and do minor cleaning
############################# 

# Initialize empty list to store panelist purchase data for each year
list_purchases <- vector("list", length(years))
names(list_purchases) <- as.character(years)

# Load in panelist purchase data as an element of the list initialized above
for(year in years)
{
  # Load in data of purchases
  file_name <- paste0("./tobacco_panelists_purchases_", year,  ".tsv") 
  dt_purchases <- fread(file_name, colClasses = list(character = "upc"))
  
  # Attach data table to the list initialized above
  list_purchases[[as.character(year)]] <- dt_purchases
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}

# Make `multi_hms` column equal to 1 when it is 0
# This simplifies consumption calculations
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt <- dt[multi == 0, multi := 1]
  
  dt
})

# Drop columns not needed
cols_to_drop <- c("upc_ver_uc", "i.upc_ver_uc", "coupon_value", "deal_flag_uc",
                  "department_code", "department_descr", 
                  "dataset_found_uc", "size1_change_flag_uc", 
                  "retailer_code", "store_code_uc", "total_spent",
                  "panel_year", "method_of_payment_cd", 
                  "Projection_Factor", "Projection_Factor_Magnet",
                  "Type_Of_Residence", "Household_Composition",
                  "Male_Head_Occupation", "Female_Head_Occupation",
                  "Scantrack_Market_Identifier_Cd",
                  "Scantrack_Market_Identifier_Desc", "DMA_Cd", "DMA_Name",
                  "Kitchen_Appliances", "TV_Items", "Household_Internet_Connection",
                  "Wic_Indicator_Current", "Wic_Indicator_Ever_Not_Current",
                  "Member_1_Employment",
                  "Member_2_Employment",
                  "Member_3_Employment",
                  "Member_4_Employment",
                  "Member_5_Employment",
                  "Member_6_Employment",
                  "Member_7_Employment")
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, (cols_to_drop) := NULL]
  
  dt
})

# Move purchase date and household code to the front of the data tables
cols_at_front <- c("household_code", "purchase_date")
list_purchases <- lapply(list_purchases, function(dt) 
{
  setcolorder(dt, cols_at_front)
  
  dt
})

# Order by purchase date and household code
cols_at_front <- c("household_code", "purchase_date")
list_purchases <- lapply(list_purchases, function(dt) 
{
  setorder(dt, purchase_date, household_code)
  dt
})

# Create month of purchase column and move it the front of the data tables
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, purchase_month := as.IDate(paste0(format(purchase_date, "%Y-%m"), "-01"))]
  setcolorder(dt, "purchase_month", after = "household_code")
  
  dt
})

# Rename panel year column and move it the front of the data tables
list_purchases <- lapply(list_purchases, function(dt) 
{
  setnames(dt, old = "Panel_Year", new = "purchase_year")
  setcolorder(dt, "purchase_year")
  
  dt
})

# Initialize flavor columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, `:=` (
    original = factor(NA, levels = c("0", "1")),
    menthol = factor(NA, levels = c("0", "1")),
    flavored = factor(NA, levels = c("0", "1")),
    menthol_or_flavored = factor(NA, levels = c("0", "1"))
  )]
  
  # Reorder columns
  setcolorder(dt, c("original", "menthol", "flavored", "menthol_or_flavored"), after = "upc")
  
  dt
})

# Create columns of binary variables to indicate categories
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, `:=` (
    cig = factor(fifelse(product_module_descr == "CIGARETTES", 1, 0), levels = c(0, 1)),
    e_cig = factor(fifelse(product_module_descr == "ELECTRONIC CIGARETTES - SMOKING", 1, 0), levels = c(0, 1)),
    cessation = factor(fifelse(product_module_descr == "ANTI-SMOKING PRODUCTS", 1, 0), levels = c(0, 1))
  )]
  
  # Reorder columns
  setcolorder(dt, c("cig", "e_cig", "cessation"), after = "upc")
  
  dt
})


#############################
# Clean cigarette purchase
# data
############################# 

# Nicotine per cig and nicotine absorbed per cig
# 12 mg of nicotine per cig implies 20 x 12 = 240 mg of nicotine per pack
# 1.25 mg of nicotine per cig implies 20 x 1.25 = 25 mg of nicotine absorbed per pack
nicotine_mg_per_cig <- 12
nicotine_mg_absorbed_per_cig <- 1.25
nicotine_mg_per_pack <- nicotine_mg_per_cig * 20
nicotine_mg_absorbed_per_pack <- nicotine_mg_absorbed_per_cig * 20

# Initialize columns 
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, `:=` (total_cigs = NA_integer_, 
             total_packs = NA_integer_,
             consumption_in_cigs = NA_integer_,
             consumption_in_packs = NA_real_, 
             nicotine_mg_consumed_in_packs = NA_real_, 
             nicotine_mg_absorbed_in_packs = NA_real_, 
             per_pack_price_paid = NA_real_)]
  
  # Reorder columns 
  setcolorder(dt, c("total_cigs", 
                    "total_packs", 
                    "consumption_in_cigs",
                    "consumption_in_packs", 
                    "nicotine_mg_consumed_in_packs", 
                    "nicotine_mg_absorbed_in_packs"), 
              after = "quantity")
  setcolorder(dt, "per_pack_price_paid", after = "total_price_paid")
  
  dt
})

# Drop CBD cigarettes
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[!(cig == 1 & grepl("HERBAL", upc_descr, ignore.case = TRUE))]
})

# # Correct toal size column for certain cartons
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     cig == 1 & multi != 1, .(upc, product_module_descr, multi, quantity, size1_amount, size1_units, upc_descr, total_price_paid)
#   ]
# )[order(upc, multi, quantity, size1_amount, upc_descr)]
# View(dt_unique)
list_purchases <- lapply(list_purchases, function(dt)
{
  dt[cig == 1 & multi == 10 & size1_amount == 200, multi := 1]

  dt
})

# For those purchases of cigarettes not divisible by 20 like a standard pack
# force them to be in packs of 20 so a pack means the same thing for everyone
# This is only a very few number of UPCs so this should not be a big deal
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[cig == 1 & (size1_amount == 10 | size1_amount == 25), size1_amount := 20]
  
  dt
})

# Fill in total packs, consumption in packs, nicotine, and per-pack price paid columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[cig == 1, total_cigs := multi * quantity * size1_amount]
  dt[cig == 1, total_packs := fifelse(total_cigs %% 20 == 0, total_cigs / 20, total_cigs / size1_amount)]
  
  # Consumption is in terms of packs, 12 mg of nicotine in a cig, and 
  # 1 mg of nicotine absorbed per cig. This implies 200 mg of nicotine per pack
  # and 20 mg of nicotine absorbed per pack.
  dt[cig == 1, `:=` (consumption_in_cigs = total_cigs,
                     consumption_in_packs = total_packs, 
                     nicotine_mg_consumed_in_packs = total_packs * nicotine_mg_per_pack,
                     nicotine_mg_absorbed_in_packs = total_packs * nicotine_mg_absorbed_per_pack
  )]
  
  # Per-pack prices
  dt[cig == 1, per_pack_price_paid := round(total_price_paid / total_packs, 2)]
  
  dt
})

# Ensure everything looks good
dt_unique <- unique(
  rbindlist(list_purchases)[
    cig == 1, .(upc, multi, quantity, size1_amount,
                total_packs, consumption_in_packs, nicotine_mg_per_pack,
                nicotine_mg_consumed_in_packs, nicotine_mg_absorbed_in_packs,
                total_price_paid, per_pack_price_paid, upc_descr)
  ]
)[order(multi, quantity, size1_amount)]
View(dt_unique)

# # Replace implausible per-pack prices with median of bottom 0.5% and top 0.5%, respectively
# dt_cig <- rbindlist(list_purchases)[cig == 1, ][order(household_code, purchase_date)]
# vals <- dt_cig[, per_pack_price_paid]
# low <- quantile(vals, probs = 0.005, na.rm = TRUE)
# high <- quantile(vals, probs = 0.995, na.rm = TRUE)
# dt_unique <- unique(
#   dt_cig[
#     per_pack_price_paid < low | per_pack_price_paid > high, 
#     .(upc, category, multi_hms, quantity, total_size_num,
#       total_packs, consumption_in_packs,
#       total_price_paid, per_pack_price_paid, product_descr)
#   ]
# )[order(per_pack_price_paid, product_descr)]
# View(dt_unique)
list_purchases <- lapply(list_purchases, function(dt) 
{
  # Preserve original per-pack prices
  dt[, per_pack_price_paid_raw := per_pack_price_paid]
  
  # Per-pack prices
  vals <- dt[cig == 1, per_pack_price_paid]
  
  # Bounds for bottom and top 0.5%
  low <- quantile(vals, probs = 0.005, na.rm = TRUE)
  high <- quantile(vals, probs = 0.995, na.rm = TRUE)
  
  # Tail medians
  median_low <- median(vals[vals <= low], na.rm = TRUE)
  median_high <- median(vals[vals >= high], na.rm = TRUE)
  print(paste0("Low Median: ", median_low))
  print(paste0("High Median: ", median_high))
  
  # Replace tails with respective medians 
  # Ensure $0.00 per-pack price paid is not used
  if (median_low != 0)
  {
    dt[cig == 1 & per_pack_price_paid <= low, per_pack_price_paid := median_low]
  }
  else
  {
    max_low <- max(vals[vals <= low], na.rm = TRUE)
    dt[cig == 1 & per_pack_price_paid <= low, per_pack_price_paid := max_low]
  }
  dt[cig == 1 & per_pack_price_paid >= high, per_pack_price_paid := median_high]
  
  # Recompute total price where per-pack price changed
  dt[cig == 1 & per_pack_price_paid != per_pack_price_paid_raw,
     total_price_paid := round(per_pack_price_paid * total_packs, 2)]
  
  # Drop temp column
  dt[, per_pack_price_paid_raw := NULL]
  
  dt
})

# # Ensure everything looks good
# dt_cig <- rbindlist(list_purchases)[cig == 1, ][order(household_code, purchase_date)]
# vals <- dt_cig[, per_pack_price_paid]
# low <- quantile(vals, probs = 0.005, na.rm = TRUE)
# high <- quantile(vals, probs = 0.995, na.rm = TRUE)
# dt_unique <- unique(
#   dt_cig[
#     per_pack_price_paid < low | per_pack_price_paid > high,
#     .(upc, multi, quantity, size1_amount,
#       total_packs, consumption_in_packs,
#       total_price_paid, per_pack_price_paid, upc_descr)
#   ]
# )[order(per_pack_price_paid, upc_descr)]
# View(dt_unique)


#############################
# Clean household
# information portion of
# the data
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
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, Household_Income := income_values[as.character(Household_Income)] ]
  
  dt
})

# Adjust male and female head age columns to reflect their actual age using the
# male and female head birth year columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, Male_Head_Age := fifelse(is.na(Male_Head_Birth), NA_integer_, purchase_year - Male_Head_Birth)]
  dt[, Female_Head_Age := fifelse(is.na(Female_Head_Birth), NA_integer_, purchase_year - Female_Head_Birth)]
  
  # Drop birth year columns
  cols_to_drop <- c("Male_Head_Birth", "Female_Head_Birth")
  dt[, (cols_to_drop) := NULL]
  
  dt
})

# Create mean head age column as the mean of the male and female head
# age columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, mean_head_age := apply(.SD, 1, function(x) mean(x, na.rm = TRUE)),
     .SDcols = c("Male_Head_Age", "Female_Head_Age")]
  
  # Reorder columns
  setcolorder(dt, "mean_head_age", after = "Female_Head_Age")
  
  dt
})

# Compute household member ages from member birth year and panel year columns
possible_members <- 7
list_purchases <- lapply(list_purchases, function(dt) 
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
list_purchases <- lapply(list_purchases, function(dt) 
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
  
  setcolorder(dt, "son_or_daughter_present", after = "Age_And_Presence_Of_Children")
  
  dt
})

# Create column if child, teen, or Young adult lives in the house
list_purchases <- lapply(list_purchases, function(dt) 
{
  # Child present in household
  dt[, child_present :=
       fifelse(
         pmin(Member_1_Age, Member_2_Age, Member_3_Age,
              Member_4_Age, Member_5_Age, Member_6_Age,
              Member_7_Age, na.rm = TRUE) <= 12 &
           pmax(Member_1_Age, Member_2_Age, Member_3_Age,
                Member_4_Age, Member_5_Age, Member_6_Age,
                Member_7_Age, na.rm = TRUE) >= 0,
         1, 0
       )
  ]
  dt[is.na(child_present), child_present := 0]
  dt[, child_count := rowSums(.SD >= 0 & .SD <= 12, na.rm = TRUE),
     .SDcols = patterns("^Member_\\d+_Age$")]
  
  # Teen present in household
  dt[, teen_present :=
       fifelse(
         pmin(Member_1_Age, Member_2_Age, Member_3_Age,
              Member_4_Age, Member_5_Age, Member_6_Age,
              Member_7_Age, na.rm = TRUE) <= 19 &
           pmax(Member_1_Age, Member_2_Age, Member_3_Age,
                Member_4_Age, Member_5_Age, Member_6_Age,
                Member_7_Age, na.rm = TRUE) >= 13,
         1, 0
       )
  ]
  dt[is.na(teen_present), teen_present := 0]
  dt[, teen_count := rowSums(.SD >= 13 & .SD <= 19, na.rm = TRUE),
     .SDcols = patterns("^Member_\\d+_Age$")]
  
  # Young adult present in household
  dt[, young_adult_present :=
       fifelse(
         pmin(Member_1_Age, Member_2_Age, Member_3_Age,
              Member_4_Age, Member_5_Age, Member_6_Age,
              Member_7_Age, na.rm = TRUE) <= 25 &
           pmax(Member_1_Age, Member_2_Age, Member_3_Age,
                Member_4_Age, Member_5_Age, Member_6_Age,
                Member_7_Age, na.rm = TRUE) >= 20,
         1, 0
       )
  ]
  dt[is.na(young_adult_present), young_adult_present := 0]
  dt[, young_adult_count := rowSums(.SD >= 20 & .SD <= 25, na.rm = TRUE),
     .SDcols = patterns("^Member_\\d+_Age$")]
  
  # Teen or Young adult present in household
  dt[, teen_or_young_adult_present :=
       fifelse(
         teen_present == 1 | young_adult_present == 1,
         1, 0
       )
  ]
  dt[is.na(teen_or_young_adult_present), teen_or_young_adult_present := 0]
  dt[, teen_or_young_adult_count := rowSums(.SD >= 13 & .SD <= 25, na.rm = TRUE),
     .SDcols = patterns("^Member_\\d+_Age$")]
  
  # Child or teen or Young adult present in household
  dt[, child_or_teen_or_young_adult_present :=
       fifelse(
         child_present == 1 | teen_present == 1 | young_adult_present == 1,
         1, 0
       )
  ]
  dt[is.na(child_or_teen_or_young_adult_present), child_or_teen_or_young_adult_present := 0]
  dt[, child_or_teen_or_young_adult_count := rowSums(.SD >= 0 & .SD <= 25, na.rm = TRUE),
     .SDcols = patterns("^Member_\\d+_Age$")]
  
  # Reorder columns
  setcolorder(dt, c("child_present", "child_count",
                    "teen_present", "teen_count",
                    "young_adult_present", "young_adult_count",
                    "teen_or_young_adult_present", "teen_or_young_adult_count",
                    "child_or_teen_or_young_adult_present", "child_or_teen_or_young_adult_count"), 
              after = "Age_And_Presence_Of_Children")
  
  # Drop age and presence of children column
  dt[, "Age_And_Presence_Of_Children" := NULL]
  
  dt
})

# Create column for full-time work status for male and female heads 
list_purchases <- lapply(list_purchases, function(dt)
{
  dt[, male_head_full_time := factor(fifelse(Male_Head_Employment == 3, 1, 0), levels = c(0, 1))]
  dt[, female_head_full_time := factor(fifelse(Female_Head_Employment == 3, 1, 0), levels = c(0, 1))]
  dt[, least_one_head_full_time := factor(fifelse((male_head_full_time == 1) | (female_head_full_time == 1), 1, 0), levels = c(0, 1))]
  
  # Reorder columns
  setcolorder(dt, "male_head_full_time", after = "Male_Head_Employment")
  setcolorder(dt, "female_head_full_time", after = "Female_Head_Employment")
  setcolorder(dt, "least_one_head_full_time", after = "female_head_full_time")
  
  # Drop male and female head employment columns
  cols_to_drop <- c("Male_Head_Employment", "Female_Head_Employment")
  dt[, (cols_to_drop) := NULL]
  
  dt
})

# Create column for college graduation status for male and female heads 
list_purchases <- lapply(list_purchases, function(dt)
{
  dt[, male_head_college := factor(fifelse(Male_Head_Education == 5 | Male_Head_Education == 6, 1, 0), levels = c(0, 1))]
  dt[, female_head_college := factor(fifelse(Female_Head_Education == 5 | Female_Head_Education == 6, 1, 0), levels = c(0, 1))]
  dt[, least_one_head_college := factor(fifelse((male_head_college == 1) | (female_head_college == 1), 1, 0), levels = c(0, 1))]
  
  # Reorder columns
  setcolorder(dt, "male_head_college", after = "Male_Head_Education")
  setcolorder(dt, "female_head_college", after = "Female_Head_Education")
  setcolorder(dt, "least_one_head_college", after = "female_head_college")
  
  # Drop male and female head employment columns
  cols_to_drop <- c("Male_Head_Education", "Female_Head_Education")
  dt[, (cols_to_drop) := NULL]
  
  dt
})

# Create marital status column to reflect if individual is married or not
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, Marital_Status := factor(fifelse(Marital_Status == 1, 1, 0), levels = c(1, 0))]
  
  dt
})

# Create columns for races 
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, white_race := factor(fifelse(Race == 1, 1, 0), levels = c(1, 0))]
  dt[, other_race := factor(fifelse((Race == 2 | Race == 3 | Race == 4), 1, 0), levels = c(1, 0))]
  
  # Reorder columns
  setcolorder(dt, c("white_race", "other_race"), after = "Race")
  
  # Drop race and hispanic origin columns
  cols_to_drop <- c("Race", "Hispanic_Origin")
  dt[, (cols_to_drop) := NULL]
  
  dt
})

# Create column for if household is in a state with flavored e-cig bans 
# Note, California (06) and Massachusetts (25) ban all flavored tobacco products
states_with_flavor_bans <- c(06, 25, 34, 36, 44, 49)
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, state_flavor_ban := factor(fifelse(Fips_State_Cd %in% states_with_flavor_bans, 1, 0))]
  
  # Reorder columns
  setcolorder(dt, "state_flavor_ban", after = "Fips_State_Cd")
  
  dt
})


#############################
# Change column names
# to match 2021-Onward
# data. This will help
# when I merge the monthly
# aggregated data
#############################

# Current column names
current_names <- c("product_module_code", "size1_amount", "size1_units", 
                   "upc_descr", "Household_Income", "Household_Size", 
                   "Male_Head_Age", "Female_Head_Age", "Marital_Status", 
                   "Panelist_ZipCd", "Fips_State_Cd", "Fips_County_Cd",
                   "Region_Cd")

# New column names
new_names <- c("product_module_code_hms", "total_size_num", "total_size_unit", 
               "product_descr", "household_income", "household_size",
               "male_head_age", "female_head_age", "marital_status",
               "panelist_zip_code", "fips_state_code", "fips_county_code",
               "region_code")

# Rename columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  setnames(dt, current_names, new_names)  
})


#############################
# Output panelist info
# to a file
#############################

# Create list containing only household info for each year
list_households <- lapply(list_purchases, function(dt)
{
  cols <- names(dt)
  index  <- match("household_income", cols)
  keep <- c("household_code", "purchase_year", cols[index:length(cols)])
  unique(dt[, ..keep], by = "household_code")
})

# Combine years from `list_households` to form a single data table
dt_households <- rbindlist(list_households)[order(household_code, purchase_year)]

# Write household information data table to a file
file_name_csv <- "../tobacco_panelists_information_2004-2020.csv"
file_name_rds <- "../tobacco_panelists_information_2004-2020.rds"
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
# Output all cleaned  
# data to a file
#############################

# Combine years from list to form a single data table
dt_full_panel <- rbindlist(list_purchases)[order(household_code, purchase_date)]

# Write the full panel data table to a file
file_name_csv <- "./tobacco_panelists_purchases_CLEANED_2004-2020.csv"
file_name_rds <- "./tobacco_panelists_purchases_CLEANED_2004-2020.rds"
fwrite(dt_full_panel, file_name_csv)
saveRDS(dt_full_panel, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds)) 
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else 
{
  cat("Error: File could not be written\n")
}







































