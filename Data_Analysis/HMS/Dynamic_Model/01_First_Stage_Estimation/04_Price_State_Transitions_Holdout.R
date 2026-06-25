################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script uses Halton draws to simulate price state-transitions for the
# holdout estimation. Uses holdout AR parameters (AR_Parameters_Holdout/) and
# holdout pricing spaces (Data_Holdout/Pricing_Spaces.csv).
# Writes Halton_Draw_Shocks.csv and Halton_Draw_Transitions.csv to Data_Holdout/.
################################################################################


#############################
# Preliminaries
#############################

# Clear environment, plot pane, and console
rm(list = ls())
graphics.off()
cat("\014")

# Set working directory to Dynamic_Model/ (same as original script)
wd <- file.path("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/",
                "4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/")
setwd(wd)

# Load packages
pacman::p_load(data.table, randtoolbox)

# Set scipen option to a high value to avoid scientific notation
options(scipen = 999)

# Increase the print limit
options(max.print = 999999)

# Increase the data table print limit
options(datatable.print.nrows = 20)

# Load holdout AR parameters
file_name_Phi       <- "./AR_Parameters_Holdout/AR_Parameters_Phi.csv"
file_name_Sigma_hat <- "./AR_Parameters_Holdout/AR_Parameters_Sigma.csv"
Phi_raw       <- fread(file_name_Phi)
Sigma_hat_raw <- as.matrix(fread(file_name_Sigma_hat))

# Clean up Sigma_hat matrix
Sigma_hat <- as.matrix(Sigma_hat_raw[, -1])
Sigma_hat <- apply(Sigma_hat, 2, as.numeric)

# Clean up Phi matrix
Phi   <- as.matrix(Phi_raw[, -1])
Phi   <- t(apply(Phi, 2, as.numeric))

# Extract AR1 intercept and slope terms
phi_0 <- Phi[1, ]
phi_1 <- Phi[2, ]

# Load holdout pricing spaces
file_name <- "./Data_Holdout/Pricing_Spaces.csv"
dt_pricing_spaces <- fread(file_name)

# Number of categories
K <- 2

# Number of Halton draws
R <- 200

# Number of possible pricing vectors
P_to_K <- (nrow(dt_pricing_spaces))^K


#############################
# All possible combinations
# in pricing space
#############################

# Quantile/percentile grid points (rows)
quantiles <- dt_pricing_spaces[, percentile]

# Price vectors by product (columns)
cig_price_quantiles  <- dt_pricing_spaces[, cig]
ecig_price_quantiles <- dt_pricing_spaces[, ecig]

# All |P|^|K| index combinations (cig x ecig)
combinations <- CJ(
  cig  = seq_along(quantiles),
  ecig = seq_along(quantiles)
)

# All |P|^|K| possible price vectors
price_vectors <- as.matrix(cbind(
  cigarette = cig_price_quantiles[combinations$cig],
  ecig      = ecig_price_quantiles[combinations$ecig]
))


#############################
# Simulate state-transitions
# using Halton draws
#############################

# Matrix to store Halton draw vector for each category
T <- array(0, dim = c(P_to_K, R, K))

# Matrix to store created random normal vectors
E <- matrix(0, nrow = R, ncol = K)

# Cholesky factor of Sigma_hat
L <- t(chol(Sigma_hat))
all.equal(L %*% t(L), Sigma_hat, check.attributes = FALSE)

# Halton draws z_r ~ N(0, I_K) using K prime bases
U <- halton(n = R, dim = K, normal = TRUE)

# Correlated shocks: eta_r := E[r, ] = L z_r ~ N(0, Sigma_hat)
E <- t(L %*% t(U))

# Simulate price transitions
for (k in seq_len(K))
{
  # Extract AR1 parameters for category k
  phi_0k <- phi_0[k]
  phi_1k <- phi_1[k]

  # Loop over all possible price vectors
  for (m in seq_len(P_to_K))
  {
    # Extract price in vector m for category k
    p_mk <- price_vectors[m, k]

    # Loop over all possible halton draws
    for (r in seq_len(R))
    {
      # Get random normal shock for price process
      eta_rk <- E[r, k]

      # T[m, r, k] = phi_0k + phi_1k * p_mk + eta_rk
      T[m, r, k] <- phi_0k + phi_1k * p_mk + eta_rk
    }
  }
}

# Convert T to a long data table
m_combinations <- rep(1:P_to_K, times = R * K)
r_combinations <- rep(rep(1:R, each = P_to_K), times = K)
k_combinations <- rep(1:K, each = P_to_K * R)
T_long <- data.table(
  m     = m_combinations,
  r     = r_combinations,
  k     = k_combinations,
  value = as.vector(T)
)

# Reshape to wide
T_wide <- dcast(
  T_long,
  m + r ~ k,
  value.var = "value"
)
setnames(T_wide, old = c("1", "2"), new = c("cig", "ecig"))


#############################
# Write results to a file
#############################

# Write correlated shocks (L * z) to Data_Holdout/
file_name <- "./Data_Holdout/Halton_Draw_Shocks.csv"
fwrite(data.table(E), file_name)

if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}

# Write predicted next period prices to Data_Holdout/
file_name <- "./Data_Holdout/Halton_Draw_Transitions.csv"
fwrite(T_wide, file_name)

if (file.exists(file_name))
{
  cat("Results have been written to\n", file_name, "\n")
} else
{
  cat("Error: File could not be written\n")
}
