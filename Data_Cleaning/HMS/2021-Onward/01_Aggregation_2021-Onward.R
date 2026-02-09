################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com 
# September 2025
#
# This script links Nielsen HMS purchase data for tobacco product to households 
# for 2021 onward and writes these panelist purchases data tables to a 
# a file for each year. Upon running this script, I will have data
# on household purchases for each year along with household information.
# Further note, the arbitrary numbers
# which I call "codes" can be found in the HMS manual.
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
# Get all tobacco product 
# info for all years 
#############################  

# Columns not needed from product attribute data
attributes_to_drop <- c("PROTEIN_GRAM", "PROTEIN_GRAM_cd", 
                        "MANUFACTURER_SUGGESTED_PRICE_CLAIM", "MANUFACTURER_SUGGESTED_PRICE_CLAIM_cd",
                        "CHOLESTEROL", "CHOLESTEROL_cd", "SODIUM_PRESENCE_CLAIM",
                        "SODIUM_PRESENCE_CLAIM_cd", "FAT_CALORIE_PER_SERVING_SIZE",
                        "FAT_CALORIE_PER_SERVING_SIZE_cd", "SUGAR_GRAM", 
                        "SUGAR_GRAM_cd", "PACKAGE_PRESENTATION_OPTION", 
                        "PACKAGE_PRESENTATION_OPTION_cd", "ADDITIONAL_INFORMATION",
                        "ADDITIONAL_INFORMATION_cd", "MANUFACTURING_PROCESS",
                        "MANUFACTURING_PROCESS_cd", "SERVING_SIZE_HOUSEHOLD",
                        "SERVING_SIZE_HOUSEHOLD_cd", "TRANS_FAT_GRAM",
                        "TRANS_FAT_GRAM_cd", "SERVING_SIZE_METRIC", "SERVING_SIZE_METRIC_cd",
                        "CALORIES_PER_SERVING_SIZE", "CALORIES_PER_SERVING_SIZE_cd",
                        "ORGANIC_CLAIM", "ORGANIC_CLAIM_cd", 
                        "GENETIC_MODIFICATION_CLAIM", "GENETIC_MODIFICATION_CLAIM_cd",
                        "TOTAL_CARBOHYDRATE_GRAM", "TOTAL_CARBOHYDRATE_GRAM_cd",
                        "TARGET_GROUP_GENDER", "TARGET_GROUP_GENDER_cd",
                        "MANUFACTURER_SUGGESTED_PRICE", "MANUFACTURER_SUGGESTED_PRICE_cd",
                        "PREPRICED", "PREPRICED_cd", "NUTRITIONAL_HEALTH_CLAIM",
                        "NUTRITIONAL_HEALTH_CLAIM_cd", "USDA_ORGANIC_SEAL", 
                        "USDA_ORGANIC_SEAL_cd", "FORMULATION", "FORMULATION_cd",
                        "STRATEGIC_INGREDIENT_PRESENCE_CLAIM", "STRATEGIC_INGREDIENT_PRESENCE_CLAIM_cd",
                        "FAT_PRESENCE_CLAIM", "FAT_PRESENCE_CLAIM_cd", "PRICE_REDUCTION",
                        "PRICE_REDUCTION_cd", "TOTAL_FAT_GRAM", "TOTAL_FAT_GRAM_cd", 
                        "SERVING_PER_CONTAINER", "SERVING_PER_CONTAINER_cd", "SATURATED_FAT_GRAM",
                        "SATURATED_FAT_GRAM_cd", "PACKAGE_GENERAL_SHAPE",
                        "PACKAGE_GENERAL_SHAPE_cd", "CLAIM", "CLAIM_cd",
                        "CALORIE_CLAIM", "CALORIE_CLAIM_cd",
                        "PACKAGE_MATERIAL_SUBSTANCE", "PACKAGE_MATERIAL_SUBSTANCE_cd",
                        "SODIUM", "SODIUM_cd", "DIETARY_FIBER", "DIETARY_FIBER_cd",
                        "DEAL_DERIVED", "DEAL_DERIVED_cd")

# Link product data to product attribute and product description data for each year
for(year in years)
{
  # Load in data of product hierarchy
  file_name <- paste0("./2021-Onward/", year, "/product_hierarchy_", year,  ".tsv")
  dt_product_hierarchy <- fread(file_name, colClasses = list(character = "upc"))
  
  # Filter out all non-tobacco products
  dt_tobacco_product_hierarchy <- dt_product_hierarchy[department_cd == 99525329]
  
  # Load in data of product attributes
  file_name <- paste0("./2021-Onward/", year, "/product_attributes_", year,  ".tsv")
  dt_product_attributes <- fread(file_name, colClasses = list(character = "upc"))
  
  # Drop attributes not needed
  dt_product_attributes[, (attributes_to_drop) := NULL]
  
  # Link product hierarchy data to product attribute data based on the UPCs
  dt_tobacco_products_v1 <- dt_tobacco_product_hierarchy[dt_product_attributes, on = .(upc), nomatch = 0]
  
  # Load in data on product descriptions
  file_name <- paste0("./2021-Onward/", year, "/product_descriptions_", year,  ".tsv")
  dt_product_descriptions <- fread(file_name, colClasses = list(character = "upc"))
  
  # Link product data to product descriptions data based on the UPCs
  dt_tobacco_products_final <- dt_tobacco_products_v1[dt_product_descriptions, on = .(upc), nomatch = 0]
  
  # Drop year columns not needed
  cols_to_drop <- c("i.year", "i.year.1")
  dt_tobacco_products_final[, (cols_to_drop) := NULL]
  
  # Write product data to a file
  file_name <- paste0("./2021-Onward/", year, "/tobacco_product_info_", year, ".tsv")
  fwrite(dt_tobacco_products_final, file_name)
  
  # Confirm results have been written to a file
  if (file.exists(file_name)) 
  {
    cat("Results have been written to", file_name, "\n")
  } else 
  {
    cat("Error: File could not be written\n")
  }
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}


############################# 
# Get only desired tobacco 
# products (cigs and e-cigs)
############################# 

# Obtain list of certain tobacco products for each year
for(year in years)
{
  # Load in data of product information
  file_name <- paste0("./2021-Onward/", year, "/tobacco_product_info_", year, ".tsv")
  dt_tobacco_products_v1 <- fread(file_name, colClasses = list(character = "upc"))
  
  # Look at categories in the tobacco department
  unique(dt_tobacco_products_v1[, .(category, category_cd)])
  
  # Filter out certain tobacco products using category code
  # I drop tobacco combination packs (99525354) because they end up just being
  # loose tobacco products after filtering out households not appearing
  # in all years in the HMS_Cleaning_2021-Onward.R script.
  # The included categories are cigarettes (99536898), 
  # non-vapor tobacco alternatives (99525434),
  # and vapor tobacco alternatives (99532606). I exclude smokeless tobacco
  # (129524349). Cessation is 99525434
  categories <- c(99536898, 99532606)
  dt_tobacco_products_v2 <- dt_tobacco_products_v1[category_cd %in% categories]
  
  # Look at unique sub-categories in chosen categories
  unique(dt_tobacco_products_v2[, .(sub_category, sub_category_cd)])
  
  # Filter out certain tobacco products using sub-category code
  # Pouches are code 99533444; other smokeless tobaco is 99534913 
  sub_categories <- c(99533757, 99532056, 99528755, 99017380)
  dt_tobacco_products_final <- dt_tobacco_products_v2[sub_category_cd %in% sub_categories]
  
  # Write final tobacco product information to a file
  file_name <- paste0("./2021-Onward/", year, "/tobacco_products_", year, ".tsv")
  fwrite(dt_tobacco_products_final, file_name)

  # Confirm results have been written to a file
  if (file.exists(file_name))
  {
    cat("Results have been written to", file_name, "\n")
  } else
  {
    cat("Error: File could not be written\n")
  }
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}


############################# 
# Link purchases for each 
# year to the tobacco 
# product info data created
# in the prior section
############################# 

# Get purchase data for only those products in `dt_products_final` for each year
for(year in years)
{
  # Load in product data that will be linked to purchase data
  file_name <- paste0("./2021-Onward/", year, "/tobacco_products_", year, ".tsv")
  dt_products <- fread(file_name, colClasses = list(character = "upc"))
  
  # Load in purchase data
  file_name <- paste0("./2021-Onward/", year, "/purchases_", year,  ".tsv")
  dt_purchases <- fread(file_name, colClasses = list(character = "upc"))

  # Only keep purchases of certain tobacco products
  dt_purchases_filtered <- dt_purchases[dt_products, on = .(upc), nomatch = 0]
  
  # Write filtered purchase data to a file 
  file_name <- paste0("./2021-Onward/", year, "/tobacco_purchases_", year, ".tsv")
  fwrite(dt_purchases_filtered, file_name)
  
  # Confirm results have been written to a file
  if (file.exists(file_name)) 
  {
    cat("Results have been written to", file_name, "\n")
  } else 
  {
    cat("Error: File could not be written\n")
  }
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}


############################# 
# Link household data to 
# tobacco purchase data 
# created in prior section 
############################# 

# Initialize empty list to store panelist purchase data for each year
list_purchases <- vector("list", length(years))
names(list_purchases) <- as.character(years)

# Match trip data to purchase data for each year to get household purchases
for(year in years)
{
  # Load in data of panelist trips
  file_name <- paste0("./2021-Onward/", year, "/trips_", year,  ".tsv") 
  dt_trips <- fread(file_name)
  
  # Load in data of purchases
  file_name <- paste0("./2021-Onward/", year, "/tobacco_purchases_", year,  ".tsv") 
  dt_purchases <- fread(file_name)
  
  # Match trip information to purchase data 
  dt_panelist_purchases <- dt_purchases[dt_trips, on = .(trip_code_uc), nomatch = 0]
  
  # Load in data for panelists
  file_name <- paste0("./2021-Onward/", year, "/panelists_", year,  ".tsv")
  dt_panelists <- fread(file_name)
  
  # Match panelist information to their purchase data
  dt_panelist_purchases_final <- dt_panelist_purchases[dt_panelists, on = .(household_code), nomatch = 0]
  
  # Attach data table to the list initialized above
  list_purchases[[as.character(year)]] <- dt_panelist_purchases_final
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}

# Combine years from list to form a single data table
dt <- rbindlist(list_purchases, fill = TRUE)[order(household_code, purchase_date)]

# Look at categories in the tobacco department
unique(dt[, .(category, category_cd)])

# Look at unique segments in chosen sub-categories
unique(dt[, .(segment, segment_cd)])

# Share of e-cig purchases that are refills or disposable
dt[segment_cd == 99535101 | segment_cd == 99526689, .N] / dt[category_cd == 99532606, .N]

# # Share of cessation purchases that are gum or lozenge
# dt[segment_cd == 99529022 | segment_cd == 99524253, .N] / dt[category_cd == 99525434, .N]

# Keep only necessary segments
segments <- c("99529481", "99535101", "99531761", "99526689", "99527147", "99524702")
dt_final <- dt[segment_cd %in% segments]


#############################
# Write purchase data
# to a file by year
############################# 

# Create new folder to contain panelist tobacco purchase data if such a folder does not already exist
directory_name <- "./2021-Onward/Tobacco_Panelists_Purchases_2021-Onward"
if (!dir.exists(directory_name)) 
{
  dir.create(directory_name)
  paste("Directory", directory_name, "has been created.")
} else
{
  paste("Directory", directory_name, "already exists.")
}

# Match trip data to purchase data for each year to get household purchases
for(yr in years)
{
  # Filter out by year
  dt_panelist_purchases_final <- dt_final[year == yr]
  
  # Write filtered purchase data to a file 
  file_name <- paste0(directory_name, "/tobacco_panelists_purchases_", yr, ".tsv")
  fwrite(dt_panelist_purchases_final, file_name)
  
  # Confirm results have been written to a file
  if (file.exists(file_name)) 
  {
    cat("Results have been written to", file_name, "\n")
  } else 
  {
    cat("Error: File could not be written\n")
  }
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", yr, "."))
}








