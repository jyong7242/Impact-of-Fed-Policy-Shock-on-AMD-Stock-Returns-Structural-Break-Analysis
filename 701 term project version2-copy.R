# ===============================
# Term Project: Structural Break and Unit Root Test on AMD Returns
# Author: Jinyan Yong
# ===============================

# ---- Step 1: Load Required Packages ----
# install.packages(c("quantmod", "tseries", "lmtest", "urca", "moments", "zoo"))
library(quantmod)
library(tseries)
library(lmtest)
library(urca)
library(moments)
library(zoo)

# ---- Step 2: Download AMD Daily Price Data ----
getSymbols("AMD", src = "yahoo", from = "2023-01-31", to = "2025-01-31")
prices <- Cl(AMD)

# ---- Step 3: Compute Log Returns ----
log_returns <- diff(log(prices))
log_returns <- na.omit(log_returns)

# ---- Step 4: Create Dummy Variable ----
dates <- index(log_returns)
break_date <- as.Date("2024-12-18")
dummy <- ifelse(dates >= break_date, 1, 0)

# ---- Step 5: Lagged Return and Interaction Term ----
lag_r <- stats::lag(log_returns, k = 1)
interaction <- dummy * lag_r

# ---- Step 6: Build Main Dataset ----
mydf <- data.frame(
  Date = dates,
  r = as.numeric(log_returns),
  lag_r = as.numeric(lag_r),
  D = dummy,
  interaction = as.numeric(interaction)
)
mydf <- na.omit(mydf)

# ---- Step 7: Structural Break Regression ----
model <- lm(r ~ lag_r + D + interaction, data = mydf)

# ---- Step 8: F-test for Structural Break ----
restricted_model <- lm(r ~ lag_r, data = mydf)

# ---- Step 9: ADF Unit Root Tests ----
adf_full <- ur.df(mydf$r, type = "drift", selectlags = "AIC")
pre_event <- mydf[mydf$D == 0, "r"]
post_event <- mydf[mydf$D == 1, "r"]
adf_pre <- ur.df(pre_event, type = "drift", selectlags = "AIC")
adf_post <- ur.df(post_event, type = "drift", selectlags = "AIC")

# ---- Step 10: Save Log Return Plot ----
# Step: Convert log_returns to data frame
logret_df <- data.frame(
  Date = index(log_returns),
  Return = as.numeric(log_returns)
)

# Step: Save log return plot with red line
png("log_returns_plot.png", width = 800, height = 600)

plot(logret_df$Date, logret_df$Return, type = "l",
     main = "AMD Log Returns (2023–2025)",
     xlab = "Date", ylab = "Log Return",
     col = "blue", lwd = 1.5)

abline(v = as.Date("2024-12-18"), col = "red", lty = 2, lwd = 2)

dev.off()

# ---- Step 11: Save Price Plot ----
# Step 1: Convert to data frame
price_df <- data.frame(
  Date = index(prices),
  Price = as.numeric(prices)
)

# Step 2: Save plot with red line
png("price_plot.png", width = 800, height = 600)

plot(price_df$Date, price_df$Price, type = "l",
     main = "AMD Stock Price (2023–2025)",
     xlab = "Date", ylab = "Adjusted Close Price (USD)",
     col = "darkgreen", lwd = 2)

abline(v = as.Date("2024-12-18"), col = "red", lty = 2, lwd = 2)  # red vertical line

dev.off()

# ---- Step 12: Volatility Structural Break Regression ----
squared_r <- mydf$r^2
lag_squared_r <- stats::lag(squared_r, k = 1)
interaction_vol <- mydf$D * lag_squared_r

vol_data <- data.frame(
  r2 = squared_r,
  lag_r2 = lag_squared_r,
  D = mydf$D,
  interaction = interaction_vol
)
vol_data <- na.omit(vol_data)

vol_model <- lm(r2 ~ lag_r2 + D + interaction, data = vol_data)

# ---- Step 13: Summary Statistics Table ----
summary_stats <- data.frame(
  Mean = c(mean(pre_event), mean(post_event)),
  SD = c(sd(pre_event), sd(post_event)),
  Skewness = c(skewness(pre_event), skewness(post_event)),
  Kurtosis = c(kurtosis(pre_event), kurtosis(post_event))
)
rownames(summary_stats) <- c("Pre-event", "Post-event")
write.csv(summary_stats, "summary_stats.csv")

# ---- Step 14: Rolling Volatility Plot ----

rolling_sd <- rollapply(log_returns, width = 20, FUN = sd, align = "right", fill = NA)


rolling_df <- data.frame(
  Date = index(rolling_sd),
  Vol = as.numeric(rolling_sd)
)

plot(rolling_df$Date, rolling_df$Vol, type = "l", col = "purple",
     main = "20-Day Rolling Volatility of AMD",
     xlab = "Date", ylab = "Rolling SD")
abline(v = as.Date("2024-12-18"), col = "red", lty = 2, lwd = 2)

png("rolling_volatility_plot.png", width = 800, height = 600)
plot(rolling_df$Date, rolling_df$Vol, type = "l", col = "purple",
     main = "20-Day Rolling Volatility of AMD",
     xlab = "Date", ylab = "Rolling SD")
abline(v = as.Date("2024-12-18"), col = "red", lty = 2, lwd = 2)
dev.off()


# ---- Step 15: Event Window Returns Plot ----
event_center <- as.Date("2024-12-18")
window_size <- 5  # ±5 days

event_idx <- which(index(log_returns) == event_center)
event_window_idx <- (event_idx - window_size):(event_idx + window_size)

event_window_idx <- event_window_idx[event_window_idx > 0 & event_window_idx <= length(log_returns)]

event_window_returns <- log_returns[event_window_idx]


png("event_window_returns_plot.png", width = 800, height = 600)
plot(index(event_window_returns), coredata(event_window_returns), type = "l",
     main = "Log Returns Around Fed Announcement (±5 Days)",
     col = "orange", lwd = 2,
     xlab = "Date", ylab = "Log Return")
abline(v = event_center, col = "red", lty = 2, lwd = 2)
dev.off()

# ---- Step 16: Export Model Results ----
sink("analysis_output.txt") 
cat("========== STRUCTURAL BREAK MODEL ==========\n")
print(summary(model))
cat("\n========== F-TEST ==========\n")
print(anova(restricted_model, model))
cat("========== ADF TEST: FULL SAMPLE ==========\n")
print(summary(adf_full))
cat("\n========== ADF TEST: PRE-SHOCK ==========\n")
print(summary(adf_pre))
cat("\n========== ADF TEST: POST-SHOCK ==========\n")
print(summary(adf_post))
cat("========== VOLATILITY STRUCTURAL BREAK MODEL ==========\n")
print(summary(vol_model))
sink()


