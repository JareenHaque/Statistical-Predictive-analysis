# Libraries
if (!require(fpp2)) install.packages("fpp2")
if (!require(tseries)) install.packages("tseries")
if (!require(forecast)) install.packages("forecast")
if (!require(ggplot2)) install.packages("ggplot2")

library(fpp2)
library(tseries)
library(forecast)
library(ggplot2)

# Loading the CSV and extracting relevant column for the time series
ts3_data <- read.csv("D:\\MSCDAD\\Stats\\CA\\ts_data\\ts3.csv")
head(ts3_data)
ts3_series_raw <- ts3_data$x  

plot(ts3_data$x, type = "l", main = "Time Series Plot", xlab = "Time", ylab = "Values")


# Checking for missing values
if (any(is.na(ts3_series_raw))) {
  ts3_series_raw <- na.omit(ts3_series_raw)  
  print("Missing values were handled by removal.")
} else {
  print("No missing values detected.")
}


##### (STL decomposition) - before differencing
if (length(ts3_series) >= frequency(ts3_series) * 2) {
  stl_decomposed_ts <- stl(ts3_series, s.window = "periodic")
  plot(stl_decomposed_ts, main = "STL Decomposed Time Series (Before Differencing)")
} else {
  print("Data is insufficient for periodic decomposition. Using a fixed seasonal window.")
  stl_decomposed_ts <- stl(ts3_series, s.window = 7)  
  plot(stl_decomposed_ts, main = "STL Decomposed Time Series (Before Differencing)")
}

# Stationarity and Differencing
adf_test <- adf.test(ts3_series)
print("ADF Test for Stationarity:")
print(adf_test)

# Differencing if non-stationary (ADF p-value > 0.05)
if (adf_test$p.value > 0.05) {
  diff_ts_series <- diff(ts3_series)
  plot(diff_ts_series, main = "Differenced Time Series", ylab = "Values", xlab = "Index")
  
  # ADF test again on differenced data
  adf_test_diff <- adf.test(diff_ts_series)
  print("ADF Test After Differencing:")
  print(adf_test_diff)
} else {
  diff_ts_series <- ts3_series  # Data is already stationary
}

# ACF and PACF Analysis for Differenced Series to identify seasonality
ggtsdisplay(diff_ts_series, main = "Differenced Series with ACF and PACF")

# Re-decompose the Differenced Series (STL decomposition)
if (length(diff_ts_series) >= frequency(diff_ts_series) * 2) {
  stl_decomposed_diff_ts <- stl(diff_ts_series, s.window = "periodic")
  plot(stl_decomposed_diff_ts, main = "STL Decomposed Differenced Time Series")
} else {
  print("Data is insufficient for periodic decomposition. Using a fixed seasonal window.")
  stl_decomposed_diff_ts <- stl(diff_ts_series, s.window = 7)  
  plot(stl_decomposed_diff_ts, main = "STL Decomposed Differenced Time Series")
}



####
# Train-Test Split for Model Validation
train_size <- length(ts3_series) - 12  
train_set <- ts3_series[1:train_size]
test_set <- ts3_series[(train_size + 1):length(ts3_series)]

# ARIMA Model on the Training Set
train_model <- auto.arima(train_set)
train_forecast <- forecast(train_model, h = 12)
####


# Fit ARIMA Model using auto.arima
auto_fit <- auto.arima(train_set)
print("Auto ARIMA Model Summary:")
summary(auto_fit)

# Manually Fit ARIMA Model, Based on ACF/PACF
manual_fit <- Arima(train_set, order = c(1, 1, 2))  
print("Manual ARIMA Model Summary:")
summary(manual_fit)


# Residuals from Auto ARIMA model
residuals_auto <- residuals(auto_fit)
plot(residuals_auto, main = "Residuals from Auto ARIMA Model", ylab = "Residuals", xlab = "Time")
abline(h = 0, col = "red", lty = 2)
# Residuals from Manual ARIMA model
residuals_manual <- residuals(manual_fit)
plot(residuals_manual, main = "Residuals from Manual ARIMA Model", ylab = "Residuals", xlab = "Time")
abline(h = 0, col = "red", lty = 2)

# ACF and PACF for Auto ARIMA residuals
par(mfrow = c(1, 2))  # Arrange plots side by side
acf(residuals_auto, main = "ACF of Auto ARIMA Residuals")
pacf(residuals_auto, main = "PACF of Auto ARIMA Residuals")

# ACF and PACF for Manual ARIMA residuals
acf(residuals_manual, main = "ACF of Manual ARIMA Residuals")
pacf(residuals_manual, main = "PACF of Manual ARIMA Residuals")


qqnorm(residuals(auto_fit), main="Auto QQ Plot")
qqline(residuals(auto_fit))

qqnorm(residuals(manual_fit),  main="Manual QQ Plot")
qqline(residuals(manual_fit))

# Forecasting using Manual ARIMA model
forecast_result_manual <- forecast(manual_fit, h = 12)
par(mfrow = c(1, 1))  # Set plot layout to one plot
plot(forecast_result_manual, main = "Manual ARIMA Forecast for Next 12 Periods", xlab = "Time", ylab = "Forecasted Values", col = "red")
abline(v = length(train_set) - 12, col = "red", lty = 2) 


# Forecasting using Auto ARIMA model
forecast_result_auto <- forecast(auto_fit, h = 12)

par(mfrow = c(1, 1))  #
plot(forecast_result_auto, main = "Auto ARIMA Forecast for Next 12 Periods", xlab = "Time", ylab = "Forecasted Values", col = "red")
abline(v = length(train_set) - 12, col = "red", lty = 2)  


# Accuracy for auto ARIMA and  manual ARIMA
accuracy(forecast_result_auto, test_set)

accuracy(forecast_result_manual, test_set)


# pick the better type
forecast_result = forecast_result_auto
sni
# (ETS)
ets_model <- ets(train_set)
summary(ets_model)

qqnorm(residuals(ets_model))
qqline(residuals(ets_model))

# Forecasting with ETS Model
ets_forecast <- forecast(train_set, h = 12)


# Plotting ETS Forecasts
par(mfrow = c(1, 1))  
plot(ets_forecast, main = "ETS Forecast for Next 12 Periods", xlab = "Time", ylab = "Forecasted Values", col = "blue")
abline(v = length(train_set) - 12, col = "blue", lty = 2) 

legend("topleft", legend = c("ARIMA", "ETS"), col = c("red", "blue"), lty = 1)

# Plotting ARIMA and ETS Forecasts Together
par(mfrow = c(1, 1))  
plot(forecast_result, main = "Combined ARIMA & ETS Forecast for Next 12 Periods", xlab = "Time", ylab = "Forecasted Values", col = "red")
lines(ets_forecast$mean, col = "blue")  # Overlay ETS forecast in blue
abline(v = length(train_set) - 12, col = "red", lty = 2) 
legend("topleft", legend = c("ARIMA", "ETS"), col = c("red", "blue"), lty = 1)


# Ljung-Box Test for White Noise Residuals
ljung_box_test_arima <- Box.test(forecast_result$residuals, lag = 12, type = "Ljung-Box")
print("Ljung-Box Test for Arima Residuals:")
print(ljung_box_test_arima)


# Ljung-Box Test for White Noise Residuals
ljung_box_test_ets <- Box.test(ets_forecast$residuals, lag = 12, type = "Ljung-Box")
print("Ljung-Box Test ETS for Residuals:")
print(ljung_box_test_ets)


#Compare ARIMA and ETS Forecast Performance
message("Comparing ARIMA vs ETS Forecast Accuracy on Test Set:")
arima_accuracy <- accuracy(forecast_result, test_set)
ets_accuracy <- accuracy(ets_forecast, test_set)

message("ARIMA Forecast Accuracy Metrics:")
print(arima_accuracy)
message("ETS Forecast Accuracy Metrics:")
print(ets_accuracy)


# ARIMA Forecast vs Actual Plot
plot(train_set, col="black", lwd=2, xlim=c(1, length(test_set) + length(train_set)),
     ylim=range(c(train_set, test_set, forecast_result$mean)), xlab="Time", ylab="Values",
     main="ARIMA Forecast vs Actual")

lines(test_set, col="black", lwd=2)  

lines(forecast_result$mean, col="red" , lwd=2, lty=2)  

abline(v=length(train_set), col="red", lty=2)


# Manual ARIMA Forecast vs Actual Plot
plot(train_set, col="black", lwd=2, xlim=c(1, length(test_set) + length(train_set)),
     ylim=range(c(train_set, test_set, forecast_result_manual$mean)), xlab="Time", ylab="Values",
     main="Manual ARIMA Forecast vs Actual")

lines(test_set, col="black", lwd=2)  

lines(forecast_result_manual$mean, col="blue", lwd=2, lty=2)  

abline(v=length(train_set), col="blue", lty=2)


# ETS Forecast vs Actual Plot
plot(train_set, col="black", lwd=2, xlim=c(1, length(test_set) + length(train_set)),
     ylim=range(c(train_set, test_set, ets_forecast$mean)), xlab="Time", ylab="Values",
     main="ETS Forecast vs Actual")

lines(test_set, col="black", lwd=2)  

lines(ets_forecast$mean, col="blue", lwd=2, lty=2)  

abline(v=length(train_set), col="blue", lty=2)


# Combined ARIMA & ETS Forecast Plot
plot(forecast_result$mean, type="l", col="red", lwd=2, xlim=c(1, length(test_set) + length(train_set)),
     ylim=range(c(train_set, test_set, forecast_result$mean, ets_forecast$mean)),
     xlab="Time", ylab="Values", main="Combined ARIMA & ETS Forecast for Next 12 Periods")

lines(ets_forecast$mean, col="blue", lwd=2, lty=2)  

lines(test_set, col="black", lwd=2)  

abline(v=length(train_set), col="red", lty=2)

legend("topleft", legend = c("ARIMA", "ETS", "Actual"), col = c("red", "blue", "black"), lty = 1)



# Model Comparison
metrics_comparison <- data.frame(
  Metric = c("RMSE", "MAE", "MAPE"),
  ARIMA = c(arima_accuracy["RMSE"], arima_accuracy["MAE"], arima_accuracy["MAPE"]),
  ETS = c(ets_accuracy["RMSE"], ets_accuracy["MAE"], ets_accuracy["MAPE"])
)

# Displaying the comparison for all metrics
print("Model Comparison Based on RMSE, MAE, and MAPE:")
print(metrics_comparison)

# Final Evaluation:Based on RMSE, MAE, or MAPE
best_model <- "ARIMA"
best_score <- Inf 

# best performing model
for (metric in c("RMSE", "MAE", "MAPE")) {
  arima_value <- metrics_comparison[metrics_comparison$Metric == metric, "ARIMA"]
  ets_value <- metrics_comparison[metrics_comparison$Metric == metric, "ETS"]
  
#Checking missing vals (NA) 
  if (!is.na(arima_value) & !is.na(ets_value)) {
    if (arima_value < ets_value) {
      if (arima_value < best_score) {
        best_model <- "ARIMA"
        best_score <- arima_value
      }
    } else if (ets_value < best_score) {
      best_model <- "ETS"
      best_score <- ets_value
    }
  } else {
    if (is.na(arima_value)) {
      best_model <- "ETS"
    } else {
      best_model <- "ARIMA"
    }
  }
}

cat("The best model is:", best_model)
