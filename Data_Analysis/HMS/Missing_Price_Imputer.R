################################################################################
# William Brasic 
# The Unvarersity of Arizona
# wbrasic97@gmail.com
# September 2025
#
# This script creates a function to impute missing prices
################################################################################


fill_missing_price <- function(dt, price_var, state_var, month_var) {
  
  median_safe <- function(x) {
    m <- median(x, na.rm = TRUE)
    if (is.na(m)) NA_real_ else m
  }
  
  # State–month median
  median_state_month <- dt[
    , .(med1 = median_safe(get(price_var))),
    by = c(state_var, month_var)
  ]
  
  # Month median
  median_month <- dt[
    , .(med2 = median_safe(get(price_var))),
    by = month_var
  ]
  
  # Merge medians
  dt[, med1 := median_state_month[.SD, on = c(state_var, month_var), med1]]
  dt[, med2 := median_month[.SD, on = month_var, med2]]
  
  # Fill missing prices
  dt[is.na(get(price_var)),
     (price_var) := fcoalesce(med1, med2)]
  
  # Cleanup
  dt[, c("med1", "med2") := NULL]
  
  # Check
  dt[is.na(get(price_var)), .N]
}
