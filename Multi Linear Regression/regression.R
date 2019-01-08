

# load required packages
library(gridExtra)
library(caret)
library(data.table)
library(dplyr)
library(readxl)
library(lmtest)
library(readr)
library(glmnet)
library(MASS)
library(GGally)


# PROBLEM 1 

# reading the data
bs_data <- read_csv("blood_sulphate.csv")
bs_data <- as.data.frame(bs_data)

bs_log_model <- lm(log(Sulfate) ~ log(Hours), data = bs_data)


# PART A 
# display in log coordinates - data points and regression line
par(mfrow = c(1, 2))
plot(log(Sulfate) ~ log(Hours), data = bs_data, col = "grey", pch = 20, cex = 1.5,
     main = "Sulfate concentration vs Hours (in log coordinates)")
abline(bs_log_model, col="darkorange", lwd=3)


# PART B 
# display in original coordinates - data points and regression curve (working code)
plot(Sulfate ~ Hours, data = bs_data, col = "grey", pch = 20, cex = 1.5,
     main = "Sulfate concentration vs Hours (in original coordinates)")
pred <- predict(bs_log_model,type="r")
lines(bs_data$Hours,exp(pred),col="darkorange",lwd=3)


# PART C 
# fitted vs residuals - log coordinates
par(mfrow = c(1, 2))
plot(fitted(bs_log_model), resid(bs_log_model), col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Fitted vs Residuals (In log coordinates)")
abline(h = 0, col ="dodgerblue", lwd = 2)


# fitted vs residuals - original coordinates
res <- bs_data$Sulfate - exp(pred)
fit <- exp(pred)

plot(fit, res, col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Fitted vs Residuals (In original coordinates)")
abline(h = 0, col ="dodgerblue", lwd = 2)


# PART D
# some r squared calculations to justify why model is good. full details are in written report
summary(bs_log_model)$r.squared
summary(bs_log_model)$adj.r.squared



# PROBLEM 2
# load the data
mass_dt <- read_csv("mass.csv")

# PART A
# build the linear model
mass_model <- lm(Mass ~ ., data = mass_dt)

#fitted vs residuals plot
plot(fitted(mass_model), resid(mass_model), col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Data from Mass Model")
abline(h = 0, col ="dodgerblue", lwd = 2)


# PART B
mass_cr_model <- lm(Mass^(1/3) ~ ., data = mass_dt)

# plot fitted vs residuals in cube root coordinates
par(mfrow = c(1, 2))
plot(fitted(mass_cr_model), resid(mass_cr_model), col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Data from Cube Root Mass Model in cube root coordinate")
abline(h = 0, col ="dodgerblue", lwd = 2)

# plot fitted vs resoduals in original coordinates
pred <- predict(mass_cr_model)
res <- mass_dt$Mass - (pred)^3
fit <- (pred)^3
plot(fit, res, col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Data from Cube Root Mass Model in original coordinate")
abline(h = 0, col ="dodgerblue", lwd = 2)


# PART C - EXPLANATION
# some calculation for r squared, adjusted r squared for justification. full details are in written report

# for simple mass model
summary(mass_model)$r.squared
summary(mass_model)$adj.r.squared


rss_mass <- c(crossprod(mass_model$residuals))
mse_mass <- rss_mass / length(mass_model$residuals)
rmse_mass <- sqrt(mse_mass)

# for cube root mass model (after back transformation)
# r squared
r2 <- cor(mass_dt$Mass, (pred)^3)^2

# adjusted r squared
n <- nrow(mass_dt)
p <- 10 # number of predictors
adjR <- 1-(1-r2) * ((n-1) / (n-p-1))

# rmse
rss_mass_cr <- c(crossprod(mass_dt$Mass - (pred)^3))
mse_mass_cr <- rss_mass_cr / length(mass_cr_model$residuals)
rmse_mass_cr <- sqrt(mse_mass_cr)
                 
summary(mass_model)
summary(mass_cr_model)

# standardized residual plots for both model for comparison
par(mfrow = c(1,1))
plot(mass_model)
plot(mass_cr_model)



# PROBLEM 3 
# PART A
# load the dataset
abalone <- read_csv("abalone.csv")

# since the age is calculated by adding 1.5 (a constant) to the response variable rings, we will add 1.5 to 
# rings variable and treat it as age in data
abalone$Rings <- abalone$Rings + 1.5

abalone_model1 <- lm(Rings ~ . - Sex, data = abalone )

par(mfrow = c(2, 2))
#fitted vs residuals
plot(fitted(abalone_model1), resid(abalone_model1), col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Data from Abalone Model (without 'Sex' predictor)")
abline(h = 0, col ="dodgerblue", lwd = 2)


# PART B
is.factor(abalone$Sex)
abalone$Sex <- as.factor(abalone$Sex)

abalone_model2 <- lm(Rings ~ ., data = abalone)
summary(abalone_model2)$r.squared

#fitted vs residuals
plot(fitted(abalone_model2), resid(abalone_model2), col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Fitted vs Residual from Abalone Model (with 'Sex' predictor)")
abline(h = 0, col ="dodgerblue", lwd = 2)


# PART C
abalone_log_model1 <- lm(log(Rings) ~ . -Sex, data = abalone)

#fitted vs residuals
pred <- predict(abalone_log_model1)
res <- abalone$Rings - exp(pred)
fit <- exp(pred)

plot(fit, res, col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Fitted vs Residual from log Abalone log Model (without Sex' Predictor)")
abline(h = 0, col ="dodgerblue", lwd = 2)


# PART D
abalone_log_model2 <- lm(log(Rings) ~ ., data = abalone)

#fitted vs residuals
pred <- predict(abalone_log_model2)
res <- abalone$Rings - exp(pred)
fit <- exp(pred)


plot(fit, res, col = 'darkorange',pch = 20,xlab = "Fitted", 
     ylab = "Residuals", main = "Fitted vs Residual from log Abalone log Model (with 'gender'Sex' predictor)")
abline(h = 0, col ="dodgerblue", lwd = 2)


# PART E
# standardized residual plots for both model for comparison
par(mfrow = c(1,1))
plot(abalone_model1)
plot(abalone_model2)
plot(abalone_log_model1)
plot(abalone_log_model2)


# pairs plot for abalone dataset
ggpairs(abalone, aes(colour = Sex, alpha = 0.8), title="Pairs plot for abalone dataset") + 
  theme_grey(base_size = 8)

# PART F
# glmnet
abalone_dm <- data.matrix(abalone)
x_var_no_gender <- abalone_dm[,2:8]
x_var_gender <- abalone_dm[,1:8]
y_var <- abalone_dm[,9]
log_y_var <- log(abalone_dm[,9]) 

# fit glm with all predictors but gender
glm_model_no_gender <- cv.glmnet(x_var_no_gender,y_var, alpha = 0)

# fit glm with all predictors 
glm_model_with_gender <- cv.glmnet(x_var,y_var, alpha = 0)


# fit glm with log response and all predictors but gender
glm_model_log_no_gender <- cv.glmnet(x_var_no_gender,log_y_var, alpha = 0)


# fit glm with log response and all predictors 
glm_model_log_with_gender <- cv.glmnet(x_var_gender,log_y_var, alpha = 0)


# plot all the glm graphs
par(mfrow = c(2, 2))
plot(glm_model_no_gender)
plot(glm_model_with_gender)
plot(glm_model_log_no_gender)
plot(glm_model_log_with_gender)

# calculate lambda values at which we get minimum cross-validated error
glm_model_no_gender$lambda.min
glm_model_with_gender$lambda.min
glm_model_log_no_gender$lambda.min
glm_model_log_with_gender$lambda.min

# min prediction error 
glm_model_no_gender$lambda.1se
glm_model_with_gender$lambda.1se
glm_model_log_no_gender$lambda.1se
glm_model_log_with_gender$lambda.1se


# calculating r squared for all glmnet models for comparison
r2_no_gender_a <- summary(abalone_model1)$r.squared
r2_gender_b <- summary(abalone_model2)$r.squared

# for log models calculate from back tranformation
pred <- predict(abalone_log_model1)
r2_log_no_gender_c <- cor(abalone$Rings,exp(pred))^2

pred <- predict(abalone_log_model2)
r2_log_gender_d <- cor(abalone$Rings,exp(pred))^2


# calculate r squared for glm models
pred <- predict(glm_model_no_gender, newx = x_var_no_gender, s = "lambda.1se")
r2_no_gender <- cor(y_var,pred)^2

pred <- predict(glm_model_with_gender, newx = x_var, s = "lambda.1se")
r2_gender <- cor(y_var,pred)^2

pred <- predict(glm_model_log_no_gender, newx = x_var_no_gender, s = "lambda.1se")
r2_log_no_gender <- cor(y_var, exp(pred))^2

pred <- predict(glm_model_log_with_gender, newx = x_var, s = "lambda.1se")
r2_log_gender <- cor(y_var, exp(pred))^2


