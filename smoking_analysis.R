# =============================================================================
# Smoking Behaviour Prediction — UK Survey Data
# =============================================================================
# A comprehensive machine learning analysis of smoking behaviour using
# UK survey data (1,691 respondents). Covers EDA, k-means clustering,
# linear regression, logistic regression, CART, Random Forest,
# and statistical diagnostics.
#
# Dataset: https://www.kaggle.com/datasets/utkarshx27/smoking-dataset-from-uk
# Author: JT
# =============================================================================


# --- 1. SETUP & DATA PREPROCESSING ------------------------------------------

library(tidyverse)
library(magrittr)
library(naniar)
library(ggplot2)
library(cluster)
library(caTools)
library(car)
library(lmtest)
library(randomForest)
library(rpart)
library(rpart.plot)

# Load dataset
data_check <- read.csv("data/smoking.csv")

# Handle missing values in smoking amount columns
data_check$amt_weekends[is.na(data_check$amt_weekends)] <- 0
data_check$amt_weekdays[is.na(data_check$amt_weekdays)] <- 0

# Convert categorical variables to factors
factor_cols <- c("gender", "marital_status", "highest_qualification",
                 "nationality", "ethnicity", "gross_income",
                 "type", "region", "smoke")
data_check[factor_cols] <- lapply(data_check[factor_cols], as.factor)

# Convert smoking amounts to integer
data_check$amt_weekends <- as.integer(data_check$amt_weekends)
data_check$amt_weekdays <- as.integer(data_check$amt_weekdays)

str(data_check)


# --- 2. EXPLORATORY DATA ANALYSIS (Bar Plots) --------------------------------

ggplot(data_check, aes(x = gender, fill = smoke)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Distribution of Smoke by Gender", x = "Gender", y = "Count")

ggplot(data_check, aes(x = marital_status, fill = smoke)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Distribution of Smoke by Marital Status", x = "Marital Status", y = "Count")

ggplot(data_check, aes(x = highest_qualification, fill = smoke)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Distribution of Smoke by Highest Qualification",
       x = "Highest Qualification", y = "Count")

ggplot(data_check, aes(x = nationality, fill = smoke)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Distribution of Smoke by Nationality", x = "Nationality", y = "Count")

ggplot(data_check, aes(x = ethnicity, fill = smoke)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Distribution of Smoke by Ethnicity", x = "Ethnicity", y = "Count")


# --- 3. UNSUPERVISED LEARNING: K-MEANS CLUSTERING ---------------------------

data_cluster <- data_check %>%
  select(age, amt_weekends, amt_weekdays) %>%
  na.omit()

scaled_data <- scale(data_cluster)

# Find optimal clusters via silhouette scores
max_clusters <- 10
silhouette_scores <- numeric(max_clusters)

for (k in 2:max_clusters) {
  kmeans_model <- kmeans(scaled_data, centers = k, nstart = 10)
  silhouette_scores[k] <- mean(silhouette(kmeans_model$cluster, dist(scaled_data)))
}

plot(2:max_clusters, silhouette_scores[2:max_clusters], type = "b",
     xlab = "Number of clusters", ylab = "Average Silhouette Score")

optimal_clusters <- which.max(silhouette_scores)
cat("Optimal number of clusters:", optimal_clusters, "\n")

# Fit final k-means
set.seed(2024)
kmeans_result <- kmeans(scaled_data, centers = optimal_clusters)
data_cluster$cluster <- kmeans_result$cluster
data_cluster$amt_total <- data_cluster$amt_weekdays + data_cluster$amt_weekends

ggplot(data_cluster, aes(x = age, y = amt_total, color = factor(cluster))) +
  geom_point() +
  labs(title = "Clusters of Age vs. Total Smoking Amount",
       x = "Age", y = "Total Smoking Amount")


# --- 4. SUPERVISED LEARNING: TRAIN-TEST SPLIT --------------------------------

set.seed(2024)

# Model 1: Reduced parameters
train_test_data <- data_check
train_test_data$smoke_binary <- ifelse(train_test_data$smoke == "Yes", 1, 0)
train_test_data <- subset(train_test_data,
                          select = c(gender, age, marital_status,
                                     highest_qualification, gross_income,
                                     smoke_binary))

split <- sample.split(train_test_data$smoke_binary, SplitRatio = 0.8)
train_data <- subset(train_test_data, split == TRUE)
test_data  <- subset(train_test_data, split == FALSE)

lm_model_tts <- lm(smoke_binary ~ ., data = train_data)
predictions <- predict(lm_model_tts, newdata = test_data)
binary_predictions <- ifelse(predictions > 0.5, 1, 0)

# Evaluation metrics
TP <- sum((test_data$smoke_binary == 1) & (binary_predictions == 1))
TN <- sum((test_data$smoke_binary == 0) & (binary_predictions == 0))
FP <- sum((test_data$smoke_binary == 0) & (binary_predictions == 1))
FN <- sum((test_data$smoke_binary == 1) & (binary_predictions == 0))

accuracy <- (TP + TN) / (TP + TN + FP + FN)
conf_matrix <- table(Actual = test_data$smoke_binary, Predicted = round(predictions))
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
rmse <- sqrt(mean((predictions - test_data$smoke_binary)^2))

cat("\n=== Model 1 (Reduced Parameters) ===\n")
cat("Accuracy:", sprintf("%.4f", accuracy), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-score:", round(f1_score, 4), "\n")
cat("RMSE:", round(rmse, 4), "\n")

# Model 2: All parameters (excluding smoking amounts)
train_test_data2 <- data_check
train_test_data2$smoke_binary <- ifelse(train_test_data2$smoke == "Yes", 1, 0)
train_test_data2 <- subset(train_test_data2,
                           select = c(gender, age, marital_status,
                                      highest_qualification, nationality,
                                      ethnicity, gross_income, region,
                                      smoke_binary))

split2 <- sample.split(train_test_data2$smoke_binary, SplitRatio = 0.8)
train_data2 <- subset(train_test_data2, split2 == TRUE)
test_data2  <- subset(train_test_data2, split2 == FALSE)

lm_model_tts2 <- lm(smoke_binary ~ ., data = train_data2)
predictions2 <- predict(lm_model_tts2, newdata = test_data2)
binary_predictions2 <- ifelse(predictions2 > 0.5, 1, 0)

TP2 <- sum((test_data2$smoke_binary == 1) & (binary_predictions2 == 1))
TN2 <- sum((test_data2$smoke_binary == 0) & (binary_predictions2 == 0))
FP2 <- sum((test_data2$smoke_binary == 0) & (binary_predictions2 == 1))
FN2 <- sum((test_data2$smoke_binary == 1) & (binary_predictions2 == 0))
accuracy2 <- (TP2 + TN2) / (TP2 + TN2 + FP2 + FN2)

cat("\n=== Model 2 (All Parameters) ===\n")
cat("Accuracy:", sprintf("%.4f", accuracy2), "\n")


# --- 5. LINEAR REGRESSION ANALYSIS -------------------------------------------

# Model with all predictors
data_simplified <- data_check
data_simplified$smoke_binary <- as.integer(data_check$smoke == "Yes")
data_simplified <- subset(data_simplified, select = -c(smoke, type, amt_weekdays, amt_weekends))

lm_model <- lm(smoke_binary ~ ., data = data_simplified)
summary(lm_model)

# Model with reduced predictors
data_simplified2 <- data_check
data_simplified2$smoke_binary <- as.integer(data_simplified2$smoke == "Yes")
data_simplified2 <- subset(data_simplified2,
                           select = -c(amt_weekends, amt_weekdays, smoke, type,
                                       ethnicity, nationality, region))

lm_model2 <- lm(smoke_binary ~ ., data = data_simplified2)
summary(lm_model2)


# --- 6. CART ANALYSIS --------------------------------------------------------

cart_model <- rpart(smoke_binary ~ ., data = train_data2, method = "class")

prp(cart_model, type = 1, extra = 1, fallen.leaves = TRUE,
    branch.lty = 2, box.col = "lightblue", split.box.col = "lightgreen", cex = 0.7)

predictions_cart <- predict(cart_model, newdata = test_data2, type = "class")
accuracy_cart <- mean(predictions_cart == test_data$smoke_binary)
cat("\nCART Accuracy:", sprintf("%.4f", accuracy_cart), "\n")


# --- 7. LOGISTIC REGRESSION --------------------------------------------------

logistic_model <- glm(smoke_binary ~ ., data = train_data, family = binomial, maxit = 500)
predictions_logit <- predict(logistic_model, newdata = test_data2, type = "response")
accuracy_logit <- mean((predictions_logit > 0.5) == test_data2$smoke_binary)
cat("Logistic Regression Accuracy:", sprintf("%.4f", accuracy_logit), "\n")


# --- 8. DIAGNOSTICS ----------------------------------------------------------

# Breusch-Pagan test for heteroscedasticity
bptest(lm_model2)

# Shapiro-Wilk test for normality of residuals
shapiro.test(resid(lm_model2))

# Cook's Distance — identify influential observations
n_influential <- length(which(cooks.distance(lm_model2) > 4 / length(cooks.distance(lm_model2))))
cat("\nInfluential observations (Cook's Distance):", n_influential, "\n")

final_data <- data_simplified2[which(
  cooks.distance(lm_model2) < 4 / length(cooks.distance(lm_model2))
), ]

# Refit model on cleaned data
final_model <- lm(smoke_binary ~ gender + age + marital_status + highest_qualification,
                   data = final_data)
par(mfrow = c(2, 2))
plot(final_model)

# Verify diagnostics on final model
bptest(final_model)
shapiro.test(resid(final_model))
summary(final_model)

# VIF — collinearity check
vif_results <- car::vif(final_model)
print(vif_results)


# --- 9. RANDOM FOREST --------------------------------------------------------

# Prepare data (remove cluster column if present)
rf_data <- data_check
if ("cluster" %in% names(rf_data)) rf_data <- subset(rf_data, select = -cluster)

rf_model <- randomForest(smoke ~ ., data = rf_data, ntree = 500)
print(rf_model)
varImpPlot(rf_model)


# --- 10. CHI-SQUARE TESTS ----------------------------------------------------

chisq_results <- list(
  `Gender vs. Smoke`       = chisq.test(table(data_check$gender, data_check$smoke)),
  `Age vs. Smoke`          = chisq.test(table(data_check$age, data_check$smoke)),
  `Marital Status vs. Smoke` = chisq.test(table(data_check$marital_status, data_check$smoke)),
  `Highest Qual vs. Smoke` = chisq.test(table(data_check$highest_qualification, data_check$smoke)),
  `Nationality vs. Smoke`  = chisq.test(table(data_check$nationality, data_check$smoke)),
  `Ethnicity vs. Smoke`    = chisq.test(table(data_check$ethnicity, data_check$smoke)),
  `Gross Income vs. Smoke` = chisq.test(table(data_check$gross_income, data_check$smoke)),
  `Region vs. Smoke`       = chisq.test(table(data_check$region, data_check$smoke))
)

p_values <- sapply(chisq_results, function(x) x$p.value)
results_table <- data.frame(Variable = names(chisq_results), P_Value = p_values)
print(results_table)


# --- 11. T-TEST --------------------------------------------------------------

age_test <- t.test(data_check$age ~ data_check$smoke)
print(age_test)
