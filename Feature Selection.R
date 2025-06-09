# CMSC 191 Project
# Feature Selection
library(readr)

# Import data
data <- read_csv("Dataset.csv")
View(data)

# Ensure binary variable is numeric (0 and 1)
data$loan_status <- as.numeric(data$loan_status)

# Pearson's Chi Squared Test of Independence

# Loop through all X variables
x_varsC <- c("education", "self_employed", "loan_term")  # Replace with your actual column names

for (var in x_varsC) {
  cat("\nChi-Squared test for", var, "vs loan_status:\n")
  table_data <- table(data[[var]], data$loan_status)  # Create contingency table
  chi_test <- chisq.test(table_data, correct = FALSE)
  print(table_data)
  print(chi_test)
}

#  Point-Biserial Correlation
# x_varsPB <- c("no_of_dependents", "income_annum", "loan_amount", "loan_term", "cibil_score",
#             "residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value")

cor.test(data$loan_status, data$no_of_dependents, method = "pearson")
cor.test(data$loan_status, data$income_annum, method = "pearson")
cor.test(data$loan_status, data$loan_amount, method = "pearson")
cor.test(data$loan_status, data$loan_term, method = "pearson")
cor.test(data$loan_status, data$cibil_score, method = "pearson")
cor.test(data$loan_status, data$residential_assets_value, method = "pearson")
cor.test(data$loan_status, data$commercial_assets_value, method = "pearson")
cor.test(data$loan_status, data$luxury_assets_value, method = "pearson")
cor.test(data$loan_status, data$bank_asset_value, method = "pearson")


