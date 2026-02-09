################################################################################
# William Brasic 
# The University of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script obtains census micro data found at 
# "https://www.census.gov/programs-surveys/acs/microdata/access.html"
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
                "4th_Year_Paper_Data/Census")
setwd(wd)

# Load packages
pacman::p_load(data.table)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 999)

# Years of census data needed
years <- 2021:2023


#############################
# Load in census housing
# micro data
############################# 

# Initialize list to store census data for each year
list_households <- vector("list", length(years))
names(list_households) <- as.character(years)

# Attach each year of the census data to the list
for (year in years)
{
  housing_zip <- paste0("https://www2.census.gov/programs-surveys/acs/data/pums/", year, "/1-Year/csv_hus.zip") 
  
  # Download data
  temp_file_housing <- tempfile(fileext = ".zip")
  download.file(housing_zip, temp_file_housing, mode = "wb", quiet = TRUE)
  
  # Unzip to temp directory
  temp_directory <- tempdir()
  unzip(temp_file_housing, exdir = temp_directory)
  
  # Read and combine both household CSVs
  dt_households <- rbindlist(list(
    fread(file.path(temp_directory, "psam_husa.csv")),
    fread(file.path(temp_directory, "psam_husb.csv"))
  ), use.names = TRUE, fill = TRUE)
  
  # Attach data table to the list initialized above
  list_households[[as.character(year)]] <- dt_households
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}

# Form data table consisting of repeated cross-section of the years
dt_households <- rbindlist(list_households, use.names = TRUE, fill = TRUE, idcol = "YEAR")

# Filter out unnecessary columns
cols_to_keep <- c("YEAR", "SERIALNO", "STATE",
                  "HHLDRAGEP", "HHLDRRAC1P", "HHT", "HINCP")
dt_households_filtered <- dt_households[, ..cols_to_keep]

# Write repeated cross-section to a file
file_name <- paste0("./census_households_2021-Onward.csv")
fwrite(dt_households_filtered, file_name)

# Confirm results have been written to a file
if (file.exists(file_name)) 
{
  cat("Results have been written to", file_name, "\n")
} else 
{
  cat("Error: File could not be written\n")
}


#############################
# Load in census persons
# micro data
############################# 

# Years of census data needed
years <- 2021:2023

# Initialize list to store census data for each year
list_persons <- vector("list", length(years))
names(list_persons) <- as.character(years)

# Attach each year of the census data to the list
for (year in years)
{
  persons_zip <- paste0("https://www2.census.gov/programs-surveys/acs/data/pums/", year, "/1-Year/csv_pus.zip") 
  
  # Download data
  temp_file_persons <- tempfile(fileext = ".zip")
  download.file(persons_zip, temp_file_persons, mode = "wb", quiet = TRUE)
  
  # Unzip to temp directory
  temp_directory <- tempdir()
  unzip(temp_file_persons, exdir = temp_directory)
  
  # Read and combine both persons CSVs
  dt_persons <- rbindlist(list(
    fread(file.path(temp_directory, "psam_pusa.csv")),
    fread(file.path(temp_directory, "psam_pusb.csv"))
  ), use.names = TRUE, fill = TRUE)
  
  # Attach data table to the list initialized above
  list_persons[[as.character(year)]] <- dt_persons
  
  # Print acknowledgement that loop is complete 
  print(paste0("Loop complete for ", year, "."))
}

# Form data table consisting of repeated cross-section of the years
dt_persons <- rbindlist(list_persons, use.names = TRUE, fill = TRUE, idcol = "YEAR")

# Filter out unnecessary columns
cols_to_keep <- c("YEAR", "SERIALNO", "STATE", "SPORDER", "AGEP", "SCHL")
dt_persons_filtered <- dt_persons[, ..cols_to_keep]

# Write repeated cross-section to a file
file_name <- paste0("./census_persons_2021-Onward.csv")
fwrite(dt_persons_filtered, file_name)

# Confirm results have been written to a file
if (file.exists(file_name)) 
{
  cat("Results have been written to", file_name, "\n")
} else 
{
  cat("Error: File could not be written\n")
}


#############################
# Merge census persons
# micro data with census
# household micro data
############################# 

# Load in census household micro data
file_name <- paste0("./census_households_2021-Onward.csv")
dt_households <- fread(file_name)

# Load in census persons micro data
file_name <- paste0("./census_persons_2021-Onward.csv")
dt_persons <- fread(file_name)

# Merge persons data onto household ata
dt_merged <- merge(
  x = dt_persons,                                          
  y = dt_households,  
  by = c("SERIALNO", "YEAR"),                                
  all.x = TRUE,                                                 
  allow.cartesian = TRUE                                    
)

# Drop duplicate state columns
dt_merged[, STATE.y := NULL]
setnames(dt_merged, "STATE.x", "STATE")

# Write merged repeated cross-section to a file
file_name <- paste0("./census_households_persons_2021-Onward.csv")
fwrite(dt_merged, file_name)

# Confirm results have been written to a file
if (file.exists(file_name)) 
{
  cat("Results have been written to", file_name, "\n")
} else 
{
  cat("Error: File could not be written\n")
}


