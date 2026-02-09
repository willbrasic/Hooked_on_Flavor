################################################################################
# William Brasic 
# The University of Arizona
# wbrasic@arizona.edu 
# williambrasic.com
# September 2025
#
# This script gets the UPC codes for certain products in group 4510 in the 
# Nielsen HMS data. Refer to the file "Product_Hierarchy_2004-2020_Data.xlsx" 
# within the "./Reference_Documentation/2004-2020_Documentation/" path
# of the working directory for further details.
################################################################################


#############################
# Preliminaries   
############################# 

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# Set working directory
setwd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/")

# Load packages
pacman::p_load(data.table)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Vector for years of data we are interested in 
years <- 2018:2020


############################# 
# Get only desired tobacco 
# products (cigs, e-cigs, 
# and cessation)
############################# 

# Link product data to product attribute and product description data for each year
for(year in years)
{
  # Load in data of products
  file_name <- paste0("./2004-2020/", year, "/products_", year,  ".tsv")
  dt_products <- fread(file_name, colClasses = list(character = "upc"))
  
  # Filter out all non-tobacco products and other certain tobacco products
  dt_products_final <- dt_products[
    product_group_code == 4510 & product_module_code %in% c(7460, 7465, 7467)
  ]
  
  # Write filtered tobacco product information to file
  file_name <- paste0("./2004-2020/", year, "/products_tobacco_", year,  ".tsv")
  fwrite(dt_products_final, file_name)
  
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

# Get purchase data for only those products in `dt_products_final`
for(year in years)
{
  # Load in product data that will be linked to purchase data
  file_name <- paste0("./2004-2020/", year, "/products_tobacco_", year,  ".tsv")
  dt_products <- fread(file_name, colClasses = list(character = "upc"))
  
  # Load in data of purchases
  file_name <- paste0("./2004-2020/", year, "/purchases_", year,  ".tsv")
  dt_purchases <- fread(file_name, colClasses = list(character = "upc"))
  
  # Only keep purchases of products in "dt_products_final"
  dt_purchases_filtered <- dt_purchases[dt_products, on = .(upc), nomatch = 0]
  
  # Write filtered purchase data to file 
  file_name <- paste0("./2004-2020/", year, "/purchases_tobacco_", year,  ".tsv")
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
  file_name <- paste0("./2004-2020/", year, "/trips_", year,  ".tsv")
  dt_trips <- fread(file_name)
  
  # Load in data of purchases
  file_name <- paste0("./2004-2020/", year, "/purchases_tobacco_", year,  ".tsv")
  dt_purchases <- fread(file_name)
  
  # Match trip information to purchase data 
  dt_panelist_purchases <- dt_purchases[dt_trips, on = .(trip_code_uc), nomatch = 0]
  
  # Load in data for panelists
  file_name <- paste0("./2004-2020/", year, "/panelists_", year,  ".tsv")
  dt_panelists <- fread(file_name)
  
  # Rename household code common to match name in `dt_panelist_purchases`
  setnames(dt_panelists, "Household_Cd", "household_code")
  
  # Match panelist information to their purchase data
  dt_panelist_purchases_final <- dt_panelist_purchases[dt_panelists, on = .(household_code), nomatch = 0]
  
  # Attach data table to the list initialized above
  list_purchases[[as.character(year)]] <- dt_panelist_purchases_final
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}

# Combine years from list to form a single data table
dt <- rbindlist(list_purchases, fill = TRUE)[order(household_code, purchase_date)]

# Move household code and purchase date to the front of the data table
setcolorder(dt, c("household_code", "purchase_date"))


#############################
# Write purchase data
# to a file by year
############################# 

# Create new folder to contain panelist tobacco purchase data if such a folder does not already exist
directory_name <- "./2004-2020/Tobacco_Panelists_Purchases_2004-2020"
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
  dt_final <- dt[Panel_Year == yr]
  
  # Write filtered purchase data to a file 
  file_name <- paste0(directory_name, "/tobacco_panelists_purchases_", yr, ".tsv")
  fwrite(dt_final, file_name)
  
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



















