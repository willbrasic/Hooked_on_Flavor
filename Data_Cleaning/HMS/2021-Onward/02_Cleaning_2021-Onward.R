################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script cleans the panelist purchase data created in the 
# 01_Aggregation_2021-Onward.R script within the same folder 
# as this current script. The arbitrary numbers used in some of the lines of 
# code come from the HMS manual.
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

# Vector for years of data we are interested in 
years <- 2021:2023


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
  dt <- dt[multi_hms == 0, multi_hms := 1]
  
  dt
})

# Drop columns not needed
cols_to_drop <- c("super_category", "super_category_cd",
                  "sub_category", "sub_category_cd",
                  "department", "department_cd",
                  "PRODUCT_SIZE", "PRODUCT_SIZE_cd",
                  "BASE_SIZE", "BASE_SIZE_cd",
                  "i.panel_year", "i.panel_year.1", "trip_code_uc", 
                  "size1_amount_hms", "size1_unit_hms", "brand_descr_hms",
                  "COMMODITY_GROUP", "COMMODITY_GROUP_cd",
                  "COMPETITIVE_CATEGORY_OGRDS", "COMPETITIVE_CATEGORY_OGRDS_cd",
                  "FORM", "FORM_cd",
                  "COMMON_CONSUMER_NAME", "COMMON_CONSUMER_NAME_cd",
                  "total_spent", "retailer_code", "store_code_uc", 
                  "method_of_payment_cd", "projection_factor", "projection_factor_magnet", 
                  "product_module_code", "multi", "start_date", "product_module_descr",
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
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, (cols_to_drop) := NULL]
  
  dt
})

# Move `purchase_date` and `household_code` to the front of the data tables
cols_at_front <- c("purchase_date", "household_code")
list_purchases <- lapply(list_purchases, function(dt) 
{
  setcolorder(dt, c(cols_at_front, setdiff(names(dt), cols_at_front)))
  
  dt
})

# Create month of purchase column and move it the front of the data tables
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, purchase_month := as.IDate(paste0(format(purchase_date, "%Y-%m"), "-01"))]
  setcolorder(dt, c("purchase_month", setdiff(names(dt), "purchase_month")))
  
  dt
})

# Rename panel year column and move it the front of the data tables
list_purchases <- lapply(list_purchases, function(dt) 
{
  setnames(dt, old = "panel_year", new = "purchase_year")
  setcolorder(dt, "purchase_year")
  
  dt
})

# Use the total size and secondary size columns to extract the 
# numerical value along with the unit of measurement
list_purchases <- lapply(list_purchases, function(dt) 
{
  # Extract numeric part 
  dt[, total_size_num := as.numeric(sub("^([0-9]*\\.?[0-9]+).*", "\\1", TOTAL_SIZE))]
  dt[, secondary_size_num := as.numeric(sub("^([0-9]*\\.?[0-9]+).*", "\\1", SECONDARY_SIZE))]
  
  # Extract the unit part
  dt[, total_size_unit := trimws(sub("^[0-9]*\\.?[0-9]+\\s*", "", TOTAL_SIZE))]
  dt[, secondary_size_unit := trimws(sub("^[0-9]*\\.?[0-9]+\\s*", "", SECONDARY_SIZE))]
  
  # Reorder columns
  setcolorder(dt, c("total_size_num", "total_size_unit"), after = "SECONDARY_SIZE_cd")
  setcolorder(dt, c("secondary_size_num", "secondary_size_unit"), after = "SECONDARY_SIZE_cd")
  
  # Drop total and secondary size columns
  dt[, `:=` (TOTAL_SIZE = NULL, TOTAL_SIZE_cd = NULL, SECONDARY_SIZE = NULL, SECONDARY_SIZE_cd = NULL)]
  
  dt
})

# # Create columns of binary variables to indicate categories
# list_purchases <- lapply(list_purchases, function(dt) 
# {
#   dt[, `:=` (
#     cig = factor(fifelse(category_cd == 99536898, 1, 0), levels = c(0, 1)),
#     pouch = factor(fifelse(category_cd == 129524349, 1, 0), levels = c(0, 1)),
#     cessation = factor(fifelse(category_cd == 99525434, 1, 0), levels = c(0, 1)),
#     ecig = factor(fifelse(category_cd == 99532606, 1, 0), levels = c(0, 1))
#   )]
#   
#   dt[, pouch_or_cessation := factor(fifelse(pouch == 1 | cessation == 1, 1, 0), levels = c(0, 1))]
#   
#   # Reorder columns
#   setcolorder(dt, c("cig", "pouch", "cessation", "pouch_or_cessation", "ecig"), after = "category_cd")
#   
#   dt
# })

# Create columns of binary variables to indicate categories
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, `:=` (
    cig       = fifelse(category_cd == 99536898, 1, 0),
    ecig      = fifelse(category_cd == 99532606, 1, 0)
  )]
  
  setcolorder(dt, c("cig", "ecig"), after = "upc")
  
  dt
})


#############################
# Clean cigarette purchase
# data
############################# 

# Nicotine per cig and nicotine absorbed per cig
# 12 mg of nicotine per cig implies 20 x 12 = 240 mg of nicotine per pack
# 1.25 mg of nicotine per cig implies 20 x 1.25 = 25 mg of nicotine absorbed per pack
nicotine_mg_consumed_per_cig <- 12
nicotine_mg_absorbed_per_cig <- 1.25
nicotine_mg_consumed_per_pack <- nicotine_mg_consumed_per_cig * 20
nicotine_mg_absorbed_per_pack <- nicotine_mg_absorbed_per_cig * 20

# Initialize columns 
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, `:=` (total_cigs = NA_integer_, 
             total_packs = NA_integer_,
             consumption_in_cigs = NA_integer_,
             consumption_in_packs = NA_real_, 
             cig_nicotine_mg_consumed = NA_real_, 
             cig_nicotine_mg_absorbed = NA_real_, 
             per_pack_price_paid = NA_real_)]
  
  # Reorder columns 
  setcolorder(dt, c("total_cigs", 
                    "total_packs", 
                    "consumption_in_cigs",
                    "consumption_in_packs", 
                    "cig_nicotine_mg_consumed", 
                    "cig_nicotine_mg_absorbed"), 
              after = "quantity")
  setcolorder(dt, "per_pack_price_paid", after = "total_price_paid")
  
  dt
})

# Drop CBD cigarettes
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[!(cig == 1 & grepl("HERBAL", product_descr, ignore.case = TRUE))]
})

# # Correct toal size column for certain cartons
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     cig == 1 & segment == "CARTON" & multi_hms == 1 & total_size_num == 20, .(upc, category, multi_hms, quantity, total_size_num, total_size_unit, product_descr, total_price_paid)
#   ]
# )[order(upc, multi_hms, quantity, total_size_num, product_descr)]
# View(dt_unique)
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[cig == 1 & segment == "CARTON" & multi_hms == 1 & total_size_num == 20, total_size_num := 200]
  
  dt
})

# # Correct the total size column for cigarettes based on UPC code lookups
# # and total price paid
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     cig == 1, .(upc, category, multi_hms, quantity, total_size_num, total_size_unit, product_descr, total_price_paid)
#   ]
# )[order(upc, multi_hms, quantity, total_size_num, product_descr)]
# View(dt_unique[total_size_num %% 20 != 0])
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[cig == 1 & upc == "1100000003", total_size_num := 20]
  dt[cig == 1 & upc == "1230010811", total_size_num := 100]
  dt[cig == 1 & upc == "2820033031", total_size_num := 100]
  dt[cig == 1 & upc == "2820033051", total_size_num := 100]
  dt[cig == 1 & upc == "4330018738", total_size_num := 20]
  dt[cig == 1 & upc == "4799520055", total_size_num := 20]
  dt[cig == 1 & upc == "68146700373", total_size_num := 100]
  dt[cig == 1 & upc == "7662277330", total_size_num := 20]
  dt[cig == 1 & upc == "80228400453", total_size_num := 100]
  dt[cig == 1 & upc == "9050001002", total_size_num := 20]
  dt[cig == 1 & upc == "9050001007", total_size_num := 20]
  
  # # For those purchases of cigarettes not divisible by 20 like a standard pack
  # # force them to be in packs of 20 so a pack means the same thing for everyone
  # # This is only a very few number of UPCs so this should not be a big deal
  dt[cig == 1 & (total_size_num == 10 | total_size_num == 25), total_size_num := 20]
  
  dt
})

# # Make `multi_hms` equal to 1 when it is 20. I believe this is an error
# # from the data.
# # Make multi_hms equal to 1 when `multi_hms` is 10 and total size column is 200 .
# # I believe this an error from the data.
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     cig == 1 & multi_hms == 20, .(upc, category, multi_hms, quantity, total_size_num, total_size_unit, total_price_paid, product_descr)
#   ]
# )[order(multi_hms, quantity, total_size_num, product_descr)]
# View(dt_unique)
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     cig == 1 & (multi_hms == 10 & total_size_num == 200), .(upc, category, multi_hms, quantity, total_size_num, total_size_unit, total_price_paid, product_descr)
#   ]
# )[order(multi_hms, quantity, total_size_num, product_descr)]
# View(dt_unique)
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[cig == 1 & multi_hms == 20, multi_hms := 10]
  dt[cig == 1 & (multi_hms == 10 & total_size_num == 200), multi_hms := 1]
  
  dt
})

# Fill in total packs, consumption in packs, nicotine, and per-pack price paid columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[cig == 1, total_cigs := multi_hms * quantity * total_size_num]
  dt[cig == 1, total_packs := fifelse(total_cigs %% 20 == 0, total_cigs / 20, total_cigs / total_size_num)]
  
  
  # Consumption is in terms of packs and nicotine consumed/absorbed
  dt[cig == 1, `:=` (consumption_in_cigs = total_cigs,
                     consumption_in_packs = total_packs,
                     cig_nicotine_mg_consumed = total_packs * nicotine_mg_consumed_per_pack,
                     cig_nicotine_mg_absorbed = total_packs * nicotine_mg_absorbed_per_pack
                     )]
  
  # Per-pack prices
  dt[cig == 1, per_pack_price_paid := round(total_price_paid / total_packs, 2)]
  
  dt
})

# # Ensure everything looks good
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     cig == 1, .(upc, category, multi_hms, quantity, total_size_num,
#                 total_packs, consumption_in_packs, nicotine_mg_consumed_per_pack,
#                 cig_nicotine_mg_consumed, cig_nicotine_mg_absorbed,
#                 total_price_paid, per_pack_price_paid, product_descr)
#   ]
# )[order(multi_hms, quantity, total_size_num)]
# View(dt_unique)

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
#     .(upc, category, multi_hms, quantity, total_size_num,
#       total_packs, consumption_in_packs,
#       total_price_paid, per_pack_price_paid, product_descr)
#   ]
# )[order(per_pack_price_paid, product_descr)]
# View(dt_unique)


#############################
# Clean e-cig purchase data
#############################

# Nicotine absorbed per mL of e-liquid (see paper Appendix for rationale)
fraction_of_nicotine_mg_absorbed_per_mL <- 0.5

# Function to convert mL of e-liquid to cigs (see paper Appendix for rationale)
mL_to_cigs <- function(conversion_factor = 0.4, nicotine_mg_consumed_per_mL, total_mL) 
{
  cigs_equivalent <- conversion_factor * nicotine_mg_consumed_per_mL * total_mL
  return(cigs_equivalent)
}

# Initialize flavor and fda authorized columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, `:=` (
    original_ecig = NA_real_,
    flavored_ecig = NA_real_,
    fda_authorized_ecig = NA_integer_
  )]
  
  setcolorder(dt, c("original_ecig", "flavored_ecig", "fda_authorized_ecig"), after = "ecig")
  
  dt
})

# Initialize other columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, `:=` (total_mL = NA_real_,
             nicotine_mg_consumed_per_mL = NA_integer_,
             ecig_nicotine_mg_consumed = NA_real_, 
             ecig_nicotine_mg_absorbed = NA_real_,
             per_mL_price_paid = NA_real_)]
  
  # Reorder columns 
  setcolorder(dt, c("total_mL", 
                    "nicotine_mg_consumed_per_mL",
                    "ecig_nicotine_mg_consumed", 
                    "ecig_nicotine_mg_absorbed"), 
              after = "cig_nicotine_mg_absorbed")
  setcolorder(dt, "per_mL_price_paid", after = "per_pack_price_paid")
  
  dt
})

# Drop starter kits (FLAVOR = "") and marijuana products
list_purchases <- lapply(list_purchases, function(dt)
{
  dt[!(ecig == 1 & (FLAVOR == "" | FLAVOR == "OG KUSH" 
                     | grepl("\\bKIT\\b", product_descr, ignore.case = TRUE)
                     | grepl("CBD", product_descr, ignore.case = TRUE) 
                     | grepl("WHISL", product_descr, ignore.case = TRUE) 
                     | grepl("CLOUD 8", product_descr, ignore.case = TRUE) 
                     | grepl("DELTA", product_descr, ignore.case = TRUE)
                     | grepl("COASTAL CLOUDS", product_descr, ignore.case = TRUE)
                     | grepl("TORCH DIAMOND", product_descr, ignore.case = TRUE)
                     | grepl("KALIBLOOM", product_descr, ignore.case = TRUE)
                     | secondary_size_unit == "GRAM"))]
  
})

# # Drop marlboro heat stick device as it does not appear to be an
# # e-cig with e-liquid
# dt_unique <- unique(rbindlist(list_purchases)[
#     ecig == 1 & grepl("MARLBORO", product_descr, ignore.case = TRUE), .(category, product_descr)
#   ]
# )[order(category, product_descr)]
# View(dt_unique)
list_purchases <- lapply(list_purchases, function(dt)
{
  dt[!(ecig == 1 & grepl("MARLBORO", product_descr, ignore.case = TRUE))]
  
})

# Drop caffeine products
list_purchases <- lapply(list_purchases, function(dt)
{
  dt[!(ecig == 1 & grepl("NU-X", product_descr, ignore.case = TRUE))]
  
})

# Correct segment and segment code for certain product
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[ecig == 1 & segment == "ELECTRONIC CIGARETTES RECHARGEABLE",
     `:=`(
       segment = "ELECTRONIC CIGARETTES DISPOSABLE",
       segment_cd = 99535101
     )]
  dt
})

# # Get flavor and product descriptions
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     ecig == 1, .(upc, FLAVOR, product_descr)
#   ]
# )[order(FLAVOR)]
# View(dt_unique)

# Correct issue where FLAVOR is NOT STATED for a certain upc, but the 
# e-cig is in fact flavored
list_purchases <- lapply(list_purchases, function(dt)
{
  dt[ecig == 1 & upc == "85003598399", flavored_ecig := 1]
  
  dt
})

# Assign flavored e-cig or not
list_purchases <- lapply(list_purchases, function(dt)
{
  dt[ecig == 1, 
     `:=` (flavored_ecig = fifelse(FLAVOR %in% c("NOT STATED", "ORIGINAL") 
                              | grepl("TOBACCO", FLAVOR, ignore.case = TRUE)
                              | grepl("REGULAR", FLAVOR, ignore.case = TRUE)
                              | grepl("CLEAR", FLAVOR, ignore.case = TRUE), 0, 1))]
  dt[ecig == 1, original_ecig := fifelse(flavored_ecig == 0, 1, 0)]
  
  dt
})

# # Ensure everything looks good
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     ecig == 1, .(upc, FLAVOR, original_ecig, flavored_ecig, product_descr)
#   ]
# )[order(FLAVOR)]
# View(dt_unique)

# Indicator FDA authorized e-cigarette purchases
list_purchases <- lapply(list_purchases, function(dt) 
{
  brands <- c("JUUL", "LOGIC (LOGICECIG)", "NJOY (NJOY)", "VUSE")
  dt[ecig == 1, fda_authorized_ecig := fifelse(BRAND %in% brands, 1, 0)]
  
  dt
})

# Convert fluid ounces to milliliters
list_purchases <- lapply(list_purchases, function(dt)
{
  # Fill in "MILLILITER" everywhere where `secondary_size_unit` is blank
  dt[ecig == 1 & secondary_size_unit == "", secondary_size_unit := "MILLILITER"]
  
  # Convert products in fluid ounces to milliliters (1 ounce = 29.57 milliliters)
  dt[ecig == 1 & secondary_size_unit == "FLUID OUNCE", secondary_size_num := secondary_size_num * 29.57]
  dt[ecig == 1 & secondary_size_unit == "FLUID OUNCE", secondary_size_unit := "MILLILITER"]
  dt[ecig == 1 & total_size_unit == "FLUID OUNCE", secondary_size_num := total_size_num * 29.57]
  dt[ecig == 1 & total_size_unit == "FLUID OUNCE", secondary_size_unit := "MILLILITER"]
  
  # Convert ounces to milliliters
  dt[ecig == 1 & secondary_size_unit == "OUNCE", secondary_size_num := secondary_size_num * 29.57]
  dt[ecig == 1 & secondary_size_unit == "OUNCE", secondary_size_unit := "MILLILITER"]
  
  # Adjust the total size column so it matches remaining observations
  # for e-cig category to avoid confusion later on
  dt[ecig == 1 & total_size_unit == "FLUID OUNCE", total_size_num := 1]
  dt[ecig == 1 & total_size_unit == "FLUID OUNCE", total_size_unit := "COUNT"]
  
  dt
})

# # Fix issue where some `multi_hms` should be 5 where `product_descr` contains "X5"
# # for certain products
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     ecig == 1 & multi_hms == 1 & grepl("X5", product_descr, ignore.case = TRUE),
#     .(upc, category, multi_hms, quantity, total_size_num,
#       total_price_paid, coupon_value, product_descr)
#   ]
# )[order(upc, product_descr)]
# View(dt_unique)
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[ecig == 1 & multi_hms == 1 & grepl("X5", product_descr, ignore.case = TRUE), multi_hms := 5]
  
  dt
})

# # Write unique e-cig product data to a file so I can obtain sizes
# # and nicotine in mg per mL. Note, `secondary_size_num` already takes
# # into account if it comes in a pack of of > 1. For instance, if
# # the e-liquid is 1mL and the `total_size_num` is 2, then
# # `secondary_size_num` would be 2mL.
# # Note, after going into excel and
# # looking up the products and filling in the .csv in excel,
# # this code block SHOULD NEVER BE RAN AGAIN until I get the data for
# # the next year.
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     ecig == 1, .(upc, segment, nicotine_mg_consumed_per_mL, secondary_size_num, secondary_size_unit, total_size_num, total_size_unit, product_descr)
#   ]
# )[order(secondary_size_num, product_descr, upc, na.last = FALSE)]
# View(dt_unique)
# file_name <- "../unique_e-cig_products_2021-Onward.csv"
# if (file.exists(file_name))
# {
#   response <- readline("File already exists. Do you wish to overwrite this file? (yes/no): ")
#   if(tolower(response) == "yes")
#   {
#     fwrite(dt_unique, file_name)
#     cat("Results have been written to", file_name, "\n")
#   } else
#   {
#     cat("Not overwriting file.")
#   }
# } else
# {
#   fwrite(dt_unique, file_name)
#   if(file.exists(file_name))
#   {
#     cat("Results have been written to", file_name, "\n")
#   } else
#   {
#     cat("Error: File could not be written\n")
#   }
# }

# Read in unique e-cig product data
file_name <- "../unique_e-cig_products_2021-Onward.csv"
dt_to_match <- fread(file_name, colClasses = list(character = "upc"))

# Match the secondary size and nicotine columns to the list of purchases
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[dt_to_match, on = "upc",
     `:=` (
       nicotine_mg_consumed_per_mL = i.nicotine_mg_per_mL,
       secondary_size_num  = i.secondary_size_num
      )]
  
  dt
})

# # Ensure everything looks good
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     ecig == 1, .(upc, segment, nicotine_mg_consumed_per_mL, secondary_size_num, secondary_size_unit, total_size_num, total_size_unit, product_descr)
#   ]
# )[order(secondary_size_num, product_descr, upc, na.last = FALSE)]
# View(dt_unique)

# Fill in total e-liquid, consumption, and nicotine columns
list_purchases <- lapply(list_purchases, function(dt) 
{
  # Notice that I don't also multiply by `total_size_num`
  # b/c `secondary_size_num` already takes that information into account
  dt[ecig == 1, total_mL := multi_hms * quantity * secondary_size_num]
  
  # Nicotine consumed and absorbed 
  dt[ecig == 1, `:=` (ecig_nicotine_mg_consumed = total_mL * nicotine_mg_consumed_per_mL,
                      ecig_nicotine_mg_absorbed = total_mL * nicotine_mg_consumed_per_mL * fraction_of_nicotine_mg_absorbed_per_mL
                      )]
  
  # Consumption in terms of cigarettes and packs of cigarettes (see paper Appendix for rationale)
  dt[ecig == 1, consumption_in_cigs := mL_to_cigs(nicotine_mg_consumed_per_mL = nicotine_mg_consumed_per_mL, total_mL = total_mL)]
  dt[ecig == 1, consumption_in_packs := consumption_in_cigs / 20]

  # Per-mL prices
  dt[ecig == 1, per_mL_price_paid := round(total_price_paid / total_mL, 2)]
  
  dt
})

# # Ensure everything looks good
# dt_unique <- unique(
#   rbindlist(list_purchases)[
#     ecig == 1 , .(upc, category, multi_hms, quantity, secondary_size_num,
#                    total_mL, consumption_in_cigs, consumption_in_packs, nicotine_mg_consumed_per_mL,
#                    ecig_nicotine_mg_consumed, ecig_nicotine_mg_absorbed,
#                    total_price_paid, per_mL_price_paid, product_descr)
#   ]
# )[order(multi_hms, quantity, secondary_size_num)]
# View(dt_unique)

# # Replace implausible per-mL prices with median of bottom 1.0% and top 1.0%, respectively
# dt_ecig <- rbindlist(list_purchases)[ecig == 1, ][order(household_code, purchase_date)]
# vals <- dt_ecig[, per_mL_price_paid]
# low <- quantile(vals, probs = 0.01, na.rm = TRUE)
# high <- quantile(vals, probs = 0.99, na.rm = TRUE)
# dt_unique <- unique(
#   dt_ecig[
#     per_mL_price_paid < low | per_mL_price_paid > high,
#     .(upc, category, multi_hms, quantity, secondary_size_num,
#       total_mL, consumption_in_packs,
#       total_price_paid, per_mL_price_paid, product_descr)
#   ]
# )[order(per_mL_price_paid, product_descr)]
# View(dt_unique)
list_purchases <- lapply(list_purchases, function(dt) 
{
  # Preserve original per-mL prices
  dt[, per_mL_price_paid_raw := per_mL_price_paid]
  
  # Per-mL prices
  vals <- dt[ecig == 1, per_mL_price_paid]
  
  # Bounds for bottom and top 1%
  low <- quantile(vals, probs = 0.01, na.rm = TRUE)
  high <- quantile(vals, probs = 0.99, na.rm = TRUE)
  
  # Tail medians
  median_low <- median(vals[vals <= low], na.rm = TRUE)
  median_high <- median(vals[vals >= high], na.rm = TRUE)
  print(paste0("Low Median: ", median_low))
  print(paste0("High Median: ", median_high))
  
  # Replace tails with respective medians 
  # Ensure $0.00 per-mL price paid is not used
  if (median_low != 0)
  {
    dt[ecig == 1 & per_mL_price_paid <= low, per_mL_price_paid := median_low]
  }
  else
  {
    max_low <- max(vals[vals <= low], na.rm = TRUE)
    dt[ecig == 1 & per_mL_price_paid <= low, per_mL_price_paid := max_low]
  }
  dt[ecig == 1 & per_mL_price_paid >= high, per_mL_price_paid := median_high]
  
  # Recompute total price where per-mL price changed
  dt[ecig == 1 & per_mL_price_paid != per_mL_price_paid_raw,
     total_price_paid := round(per_mL_price_paid * total_mL, 2)]
  
  # Drop temp column
  dt[, per_mL_price_paid_raw := NULL]
  
  dt
})

# # Ensure everything looks good
# dt_ecig <- rbindlist(list_purchases)[ecig == 1, ][order(household_code, purchase_date)]
# vals <- dt_ecig[, per_mL_price_paid]
# low <- quantile(vals, probs = 0.01, na.rm = TRUE)
# high <- quantile(vals, probs = 0.99, na.rm = TRUE)
# dt_unique <- unique(
#   dt_ecig[
#     per_mL_price_paid < low | per_mL_price_paid > high,
#     .(upc, category, multi_hms, quantity, secondary_size_num,
#       total_mL, consumption_in_packs,
#       total_price_paid, per_mL_price_paid, product_descr)
#   ]
# )[order(per_mL_price_paid, product_descr)]
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
  dt[, household_income := income_values[as.character(household_income)] ]
  
  dt
})

# # Get number of households without a male or female head (returns zero for all years as expected)
# lapply(list_purchases, function(dt)
#   nrow(dt[male_head_age == 0 & female_head_age == 0])
# )

# Adjust male and female head age columns to reflect their actual age using the
# male and female head birth year columns
list_purchases <- lapply(list_purchases, function(dt) 
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
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, mean_head_age := apply(.SD, 1, function(x) mean(x, na.rm = TRUE)),
     .SDcols = c("male_head_age", "female_head_age")]
  
  # Reorder columns
  setcolorder(dt, "mean_head_age", after = "female_head_age")
  
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
  
  setcolorder(dt, "son_or_daughter_present", after = "age_and_presence_of_children")
  
  dt
})

# Create column if child, teen, or young adult lives in the house (including HoHs)
list_purchases <- lapply(list_purchases, function(dt) 
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
list_purchases <- lapply(list_purchases, function(dt)
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
list_purchases <- lapply(list_purchases, function(dt)
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
# lapply(list_purchases, function(dt) unique(dt[, marital_status]))

# Create marital status column to reflect if individual is married or not
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, marital_status := fifelse(
    is.na(marital_status), 0,
    fifelse(marital_status == 1, 1, 0)
  )]
  
  dt
})

# Create columns for races 
list_purchases <- lapply(list_purchases, function(dt) 
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
list_purchases <- lapply(list_purchases, function(dt) 
{
  dt[, state_flavor_ban := fifelse(
    fips_state_code %in% states_with_flavor_bans, 1, 0
  )]
  
  setcolorder(dt, "state_flavor_ban", after = "fips_state_code")
  
  dt
})

# Create list containing only household info for each year
list_households <- lapply(list_purchases, function(dt)
{
  cols <- names(dt)
  index  <- match("household_income", cols)
  keep <- c("household_code", "purchase_year", cols[index:length(cols)])
  unique(dt[, ..keep], by = "household_code")
})

# Combine years from `list_households` to form a single data table
# This contains those household-years in which the household made either a 
# cig or e-cig purchase in that year
dt_households <- rbindlist(list_households)[order(household_code, purchase_year)]

# Write household information data table to a file
file_name_csv <- "../tobacco_panelists_information_2021-Onward.csv"
file_name_rds <- "../tobacco_panelists_information_2021-Onward.rds"
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

# Combine all purchases across years
dt_all_purchases <- rbindlist(list_purchases)

# Find households who ever bought an ecig (across all years)
hh_ever_ecig <- dt_all_purchases[ecig == 1, unique(household_code)]

# Find households who ever bought a cig (across all years)
hh_ever_cig <- dt_all_purchases[cig == 1, unique(household_code)]

# Cig-only households: bought cig but never bought ecig across all years
hh_cig_only <- setdiff(hh_ever_cig, hh_ever_ecig)

# Extract household info for cig-only households
cols <- names(dt_all_purchases)
index <- match("household_income", cols)
keep <- c("household_code", "purchase_year", cols[index:length(cols)])
dt_cig_households <- unique(dt_all_purchases[household_code %in% hh_cig_only, ..keep],
                            by = c("household_code", "purchase_year"))[order(household_code, purchase_year)]

# Write cig-only household information data table to a file
file_name_csv <- "../cig_only_panelists_information_2021-Onward.csv"
file_name_rds <- "../cig_only_panelists_information_2021-Onward.rds"
fwrite(dt_cig_households, file_name_csv)
saveRDS(dt_cig_households, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds))
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else
{
  cat("Error: File could not be written\n")
}

# Ecig-only households: bought ecig but never bought cig across all years
hh_ecig_only <- setdiff(hh_ever_ecig, hh_ever_cig)

# Extract household info for ecig-only households
dt_ecig_households <- unique(dt_all_purchases[household_code %in% hh_ecig_only, ..keep],
                             by = c("household_code", "purchase_year"))[order(household_code, purchase_year)]

# Write ecig-only household information data table to a file
file_name_csv <- "../ecig_only_panelists_information_2021-Onward.csv"
file_name_rds <- "../ecig_only_panelists_information_2021-Onward.rds"
fwrite(dt_ecig_households, file_name_csv)
saveRDS(dt_ecig_households, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds))
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else
{
  cat("Error: File could not be written\n")
}

# Dual purchaser households: bought both cigs and ecigs across all years
hh_both <- intersect(hh_ever_cig, hh_ever_ecig)

# Extract household info for dual purchaser households
dt_both_households <- unique(dt_all_purchases[household_code %in% hh_both, ..keep],
                             by = c("household_code", "purchase_year"))[order(household_code, purchase_year)]

# Write dual purchaser household information data table to a file
file_name_csv <- "../cig_ecig_both_panelists_information_2021-Onward.csv"
file_name_rds <- "../cig_ecig_both_panelists_information_2021-Onward.rds"
fwrite(dt_both_households, file_name_csv)
saveRDS(dt_both_households, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds))
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else
{
  cat("Error: File could not be written\n")
}


#############################
# Output cleaned cig data
# to a file
#############################

# Combine cig purchases across years to form a single data table
dt_cig <- rbindlist(list_purchases)[cig == 1, ][order(household_code, purchase_date)]
cols_to_drop <- c("ecig", "total_mL",
                  "nicotine_mg_consumed_per_mL", "ecig_nicotine_mg_consumed", "ecig_nicotine_mg_absorbed",
                  "per_mL_price_paid")
dt_cig[, (cols_to_drop) := NULL]

# Write cig purchases data to a file
file_name_csv <- "./cig_panelists_purchases_CLEANED_2021-Onward.csv"
file_name_rds <- "./cig_panelists_purchases_CLEANED_2021-Onward.rds"
fwrite(dt_cig, file_name_csv)
saveRDS(dt_cig, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds)) 
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else 
{
  cat("Error: File could not be written\n")
}


#############################
# Output cleaned e-cig data 
# to a file
#############################

# Combine e-cig purchases across years to form a single data table
dt_ecig <- rbindlist(list_purchases)[ecig == 1, ][order(household_code, purchase_date)]
cols_to_drop <- c("cig", "total_cigs", "total_packs",
                  "cig_nicotine_mg_consumed", "cig_nicotine_mg_absorbed",
                  "per_pack_price_paid")
dt_ecig[, (cols_to_drop) := NULL]

# Write e-cig purchases data to a file
file_name_csv <- "./ecig_panelists_purchases_CLEANED_2021-Onward.csv"
file_name_rds <- "./ecig_panelists_purchases_CLEANED_2021-Onward.rds"
fwrite(dt_ecig, file_name_csv)
saveRDS(dt_ecig, file_name_rds)

# Confirm results have been written to a file
if (file.exists(file_name_csv) & file.exists(file_name_rds)) 
{
  cat("Results have been written to\n", file_name_csv, "\nand\n", file_name_rds, "\n")
} else 
{
  cat("Error: File could not be written\n")
}


#############################
# Output cleaned cig 
# and e-cig data to a file
#############################

# Combine cig and e-cig purchases across years to form a single data table
dt_cig_ecig <- rbindlist(list_purchases)[cig == 1 | ecig == 1, ][order(household_code, purchase_date)]

# Write cig and e-cig purchases data to a file
file_name_csv <- "./cig_ecig_panelists_purchases_CLEANED_2021-Onward.csv"
file_name_rds <- "./cig_ecig_panelists_purchases_CLEANED_2021-Onward.rds"
fwrite(dt_cig_ecig, file_name_csv)
saveRDS(dt_cig_ecig, file_name_rds)

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
file_name_csv <- "./tobacco_panelists_purchases_CLEANED_2021-Onward.csv"
file_name_rds <- "./tobacco_panelists_purchases_CLEANED_2021-Onward.rds"
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

















