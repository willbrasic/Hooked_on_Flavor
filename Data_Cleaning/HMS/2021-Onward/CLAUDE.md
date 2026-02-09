# HMS 2021-Onward Data Cleaning Pipeline

Nielsen Homescan Panel data cleaning for tobacco purchases (cigarettes and e-cigarettes), 2021-2023.

## Scripts

### 01_Aggregation_2021-Onward.R
Links raw HMS data files to create tobacco purchase datasets.
- Filters product hierarchy to tobacco department (code 99525329)
- Keeps cigarettes (99536898) and vapor alternatives (99532606)
- Joins product info -> purchases -> trips -> panelists
- Output: `tobacco_panelists_purchases_{year}.tsv` per year

### 02_Cleaning_2021-Onward.R
Main cleaning script for cigarette and e-cigarette purchases.
- **Cigarettes**: Total packs, nicotine (12mg consumed/1.25mg absorbed per cig), per-pack prices, 0.5% tail outlier replacement
- **E-cigarettes**: Unit conversion to mL, flavored vs original classification, FDA-authorized brands (JUUL, LOGIC, NJOY, VUSE), nicotine metrics, 1% tail outlier replacement
- **Households**: Income midpoints from codes, head ages from birth years, child/teen/young adult indicators, employment/education/race/marital status, state flavor ban indicator
- Output: `cig_panelists_purchases_CLEANED_*.rds`, `ecig_panelists_purchases_CLEANED_*.rds`, `tobacco_panelists_information_*.rds`

### 03_Cleaning_Non-Tobacco_Household_Info_2021-Onward.R
Cleans household demographics for non-tobacco users (control group).
- Same demographic transformations as script 02
- Excludes households appearing in tobacco purchase data
- Output: `non-tobacco_panelists_information_2021-Onward.rds`

### 04_Monthly_Aggregation_2021-Onward.R
Aggregates cleaned data to monthly household-level panel.
- Sums consumption (packs, mL, nicotine) by household-month
- Creates complete month grid per household-year
- Mutually exclusive categories: cig only, ecig only, cig+ecig, outside option
- Real prices via tobacco CPI from FRED (series CUSR0000SEGA)
- `teen_or_young_adult_ever` and `teen_or_young_adult_always` indicators
- Output: `tobacco_panelists_purchases_monthly_CLEANED_2021-Onward.rds`

## Key Codes (from HMS manual)
- Department 99525329: Tobacco
- Category 99536898: Cigarettes
- Category 99532606: Vapor tobacco alternatives
- Segments: 99529481, 99535101 (disposable), 99531761, 99526689 (refills), 99527147, 99524702
