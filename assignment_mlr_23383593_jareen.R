# Check and install necessary packages if not already installed

if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("car", quietly = TRUE)) install.packages("car")
if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")
if (!requireNamespace("DMwR2", quietly = TRUE)) install.packages("DMwR2")
if (!requireNamespace("MLmetrics", quietly = TRUE)) install.packages("MLmetrics")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("lmtest", quietly = TRUE)) install.packages("lmtest")

# Load necessary libraries
library(ggplot2)
library(car)
library(MASS)
library(DMwR2)
library(MLmetrics)
library(caret)
library(lmtest)

# Load the dataset
mlr3_data <- read.csv("D:\\MSCDAD\\Stats\\CA\\mlr_data\\mlr3.csv")

# Data overview
head(mlr3_data)
str(mlr3_data)
names(mlr3_data)
summary(mlr3_data)


# Exploratory plots
par(mfrow = c(2, 2))
boxplot(mlr3_data$y, col = "blue", main = "Boxplot of y")
boxplot(mlr3_data$x1, col = "green", main = "Boxplot of x1")
boxplot(mlr3_data$x2, col = "yellow", main = "Boxplot of x2")

# Histograms for variables
par(mfrow = c(2, 2)) 
hist(mlr3_data$y, col = "Purple", main = "Distribution of Target (y)", xlab = "y")
hist(mlr3_data$x1, col = "Green", main = "Distribution of x1", xlab = "x1")
hist(mlr3_data$x2, col = "Yellow", main = "Distribution of x2", xlab = "x2")
barplot(height = table(mlr3_data$x3), col = "red", main = "Barplot of x3", xlab = "x3")

# Correlation matrix on original data
correlation_matrix <- cor(mlr3_data[, sapply(mlr3_data, is.numeric)])
cat("Correlation Matrix:\n")
print(correlation_matrix)


# Check if any missing values
missing_values <- is.na(mlr3_data)
missing_count <- colSums(missing_values)
print(missing_count)


# Ensuring 'x3' is a categorical variable
mlr3_data$x3 <- as.factor(mlr3_data$x3)


# Scaling the numeric variables 
mlr3_data_scaled <- mlr3_data
mlr3_data_scaled[, c("x1", "x2", "y")] <- scale(mlr3_data[, c("x1", "x2", "y")])


# Split dataset into training and testing sets
set.seed(23383593) 
train_indices <- sample(1:nrow(mlr3_data), ceiling(0.8 * nrow(mlr3_data)))
train <- mlr3_data[train_indices, ]
test <- mlr3_data[-train_indices, ]

train$x3 <- as.factor(train$x3)
test$x3 <- as.factor(test$x3)

# Cross-validation setup (k-fold cross-validation)
train_control <- trainControl(method = "cv", number = 10) 
cv_model <- train(y ~ x1 + x2 + x3, data = train, method = "lm", trControl = train_control)
print(cv_model)

# Initial multiple linear regression model
lmfit <- lm(y ~ x1 + x2 + x3, data = train)
summary(lmfit)

# Diagnostic plots for lmfit
par(mfrow = c(2, 2))
plot(lmfit)

# R-squared and Adjusted R-squared
cat("R-squared:", summary(lmfit)$r.squared, "\n")
cat("Adjusted R-squared:", summary(lmfit)$adj.r.squared, "\n")


# Example prediction
new_data <- data.frame(x1 = 45, x2 = 230, x3 = factor("A", levels = levels(mlr3_data$x3)))
predicted_y <- predict(lmfit, new_data)
cat("Predicted y for Example Data:", predicted_y, "\n")


# Exploratory plots
par(mfrow = c(2, 2))
boxplot(train$y, col = "blue", main = "Boxplot of y")
boxplot(train$x1, col = "green", main = "Boxplot of x1")
boxplot(train$x2, col = "yellow", main = "Boxplot of x2")
barplot(height = table(train$x3), col = "red", main = "Barplot of x3", xlab = "x3")

# Histograms for variables
par(mfrow = c(2, 4)) 
hist(train$y, col = "Purple", main = "Distribution of Target (y)", xlab = "y")
hist(train$x1, col = "Green", main = "Distribution of x1", xlab = "x1")
hist(train$x2, col = "Yellow", main = "Distribution of x2", xlab = "x2")

# Correlation for numeric variables
correlation_matrix <- cor(train[, sapply(train, is.numeric)])
cat("Correlation Matrix:\n")
print(correlation_matrix)

# Scatter plot matrix for numeric variables
pairs(train[, sapply(train, is.numeric)])

# Build a refined model 
refined_model_1 <- lm(y ~ x1 + x2, data = train)
summary(refined_model_1)

# Diagnostic checks for the refined model
par(mfrow = c(2, 2))
plot(refined_model_1)

# The error term has mean 0 or very close to 0
cat("Mean of Residuals (Refined Model):", mean(refined_model_1$residuals), "\n")

# Check for no multicollinearity using VIF
cat("VIF Values (Refined Model):\n")
print(vif(refined_model_1))


# Normality of residuals
shapiro_test <- shapiro.test(residuals(refined_model_1))
cat("Shapiro-Wilk Test for Normality of Residuals:\n")
print(shapiro_test)

# Q-Q Plot for Normality of Residuals
qqnorm(residuals(refined_model_1), main = "Q-Q Plot of Residuals")
qqline(residuals(refined_model_1), col = "red")


# Plot Residuals vs Fitted Values to check indpendency of residuals or auto correlation
plot(fitted(refined_model_1), residuals(refined_model_1),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residual Plot",
     pch = 20, col = "blue", cex = 0.7)

# Add a horizontal line at y = 0
abline(h = 0, col = "red", lwd = 2, lty = 2)  # Dashed red line at y = 0



######
# Apply log transformation to independent variables (x1 and x2)
train$log_x1 <- log(train$x1)
train$log_x2 <- log(train$x2)

# Fit the model with the log-transformed x1 and x2
refined_model_log_x <- lm(y ~ log_x1 + log_x2, data = train)
summary(refined_model_log_x)

# Diagnostic checks for the refined model with log(x1) and log(x2)
par(mfrow = c(2, 2))
plot(refined_model_log_x)

# Mean of residuals for log-transformed model
cat("Mean of Residuals (Log-transformed Model):", mean(refined_model_log_x$residuals), "\n")


model_without_influential <- update(refined_model_log_x, subset = !(rownames(data) %in% c(382, 432, 937)))
summary(refined_model_log_y)
summary(model_without_influential)


# Q-Q Plot for Normality of Residuals
qqnorm(residuals(refined_model_log_x), main = "Q-Q Plot of Residuals")
qqline(residuals(refined_model_log_x), col = "red")

hist(residuals(refined_model_log_x), main="Residuals Histogram", xlab="Residuals")

######





######
# Apply log transformation to the dependent variable
train$log_y <- log(train$y)


# Fit the model with the log-transformed y
refined_model_log_y <- lm(log_y ~ x1 + x2, data = train)
summary(refined_model_log_y)

# Diagnostic checks for the refined model with log(y)
par(mfrow = c(2, 2))
plot(refined_model_log_y)

# Mean of residuals for log-transformed model
cat("Mean of Residuals (Log-transformed Model):", mean(refined_model_log_y$residuals), "\n")

model_without_influential <- update(refined_model_log_y, subset = !(rownames(data) %in% c(274, 432, 792)))
summary(refined_model_log_y)
summary(model_without_influential)


# Q-Q Plot for Normality of Residuals
qqnorm(residuals(refined_model_log_y), main = "Q-Q Plot of Residuals")
qqline(residuals(refined_model_log_y), col = "red")

hist(residuals(refined_model_log_y), main="Residuals Histogram", xlab="Residuals")

# Check for no multicollinearity using VIF
cat("VIF Values (refined_model_log_y):\n")
print(vif(refined_model_log_y))

#####




# 
# Durbin-Watson Test for Independence of Residuals
dw_test <- durbinWatsonTest(refined_model_log_y)
cat("Durbin-Watson Test:\n")
print(dw_test)


# Cook's Distance for Influential Points
cooksD_refined <- cooks.distance(refined_model_log_y)
plot(cooksD_refined, type = "h", main = "Cook's Distance (Refined Model)",
     xlab = "Observation", ylab = "Cook's Distance")
abline(h = 1, col = "red", lty = 2)
cat("Influential Points (Refined Model, Cook's Distance > 1):\n")
print(which(cooksD_refined > 1))

# Check for homoscedasticity
plot(refined_model_1, which = 3)


# 
# # Box-Cox transformation to find the best lambda, which was ignored later
boxcox_model <- boxcox(refined_model_log_y, lambda = seq(-1, 2, 0.1))
best_lambda <- boxcox_model$x[which.max(boxcox_model$y)]  # Find the best lambda
print(best_lambda)
# # 
# # # Apply the best lambda from the Box-Cox transformation
# train$new_y <- train$y^best_lambda
# test$new_y <- test$y^best_lambda
# # 
# # # Fit the transformed model with Box-Cox transformation
# transformed_model <- lm(new_y ~ x1 + x2, data = train)
# summary(transformed_model)
# hist(residuals(transformed_model), main="Residuals Histogram", xlab="Residuals")
# 
# 
# # # Predict on the test set using the transformed model
# test$new_y <- test$y^best_lambda
# test$predicted_new_y <- predict(transformed_model, newdata = test)
# 
# # # Back-transform predictions
# test$predicted_y <- test$predicted_new_y^(1/best_lambda)


#####

test$log_y <- log(test$y)

predicted_values <- predict(refined_model_log_y, newdata = test)
test$predicted_y <- predicted_values

comparison_df <- data.frame(Actual = test$log_y, Predicted = predicted_values)


ggplot(comparison_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +   
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") + 
  labs(title = "Actual vs Predicted Values",
       x = "Actual log(Y)",
       y = "Predicted log(Y)") +
  theme_minimal()

residuals <- comparison_df$Actual - comparison_df$Predicted

ggplot(comparison_df, aes(x = Predicted, y = residuals)) +
  geom_point(color = "blue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Predicted Values",
       x = "Predicted log(Y)",
       y = "Residuals") +
  theme_minimal()


# Evaluate the model
evaluation_metrics <- postResample(pred = test$predicted_y, obs = test$y)
cat("Evaluation Metrics on Test Data:\n")
print(evaluation_metrics)

plot(test$y, test$predicted_y, main = "Predicted vs Actual", 
     xlab = "Actual Values", ylab = "Predicted Values", col = "blue")
abline(a = 0, b = 1, col = "red")  

train_control <- trainControl(method = "cv", number = 10)
cross_val_model <- train(y ~ x1 + x2, data = train, method = "lm", trControl = train_control)
print(cross_val_model)




