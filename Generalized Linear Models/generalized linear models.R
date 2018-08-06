
# HW6

# references used for this exercise:
# http://www4.stat.ncsu.edu/~post/josh/LASSO_Ridge_Elastic_Net_-_Examples.html
# https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html


# loading necessary packages for homework
library(glmnet)
library(MASS)
library(readxl)
library(caret)
library(boot)


# PROBLEM - 1
# load the dataset
music_data <- read.table("default_plus_chromatic_features_1059_tracks.txt",sep = ",")

#Index <- createDataPartition(c(datafr))
#train <- dataFrame[Index,]
#test <- dataFrame[-Index,]

num_features <- dim(music_data)[2]
num_examples <- dim(music_data)[1]

x_features <- music_data[,1:(num_features-2)]
x_features <- as.matrix(x_features)

# scaling of lattitude and longitude for box cox transformation 
lat <- music_data[,(num_features - 1)] + 90
long <- music_data[,(num_features)] + 180


# PART 1
# creating linear model with latitude as response
lat_model <- lm(lat ~ x_features)
lat_r2 <- summary(lat_model)$r.squared
lat_r2
mean(lat_model$residuals^2)


par(mfrow=c(2,2))
plot(lat_model)

long_model <- lm(long ~ x_features)
long_r2 <- summary(long_model)$r.squared
long_r2 
mean(long_model$residuals^2)

par(mfrow=c(2,2))
plot(long_model)


# glmnet models for comparison with regularized model using cross validation
# NOTE: we are using cv.glmnet but have set lambda values to extremely low so that the regularization effect can be 
# neglected and this model can be compared with regularized model for comparison.
lat_glm_model <- cv.glmnet(x_features, lat, alpha = 0, lambda = c(1e-9,1e-8))
min(lat_glm_model$cvm)

long_glm_model <- cv.glmnet(x_features, long, alpha = 0, lambda = c(1e-9,1e-8))
min(long_glm_model$cvm)



# PART 2
# BOX COX Tranformations
# for lattitude model
boxcox(lat_model,lambda = seq(2, 4, 1/10)) # boxcox yields lambda value of approx 3.6

boxcox_lat_model <- lm(lat^3.6 ~ x_features)
boxcox_lat_r2 <- summary(boxcox_lat_model)$r.squared
boxcox_lat_r2

# transforming back residuals for comparison
pred <- predict(boxcox_lat_model)
res <- lat - (pred)^(1/3.6)
mean(res^2) #mse

par(mfrow=c(2,2))
plot(boxcox_lat_model)

# for longitude
boxcox(long_model, lambda = seq(0,2,1/10)) #boxcox yields lambda value of approx 1.1
boxcox_long_model <- lm(long^1.1 ~ x_features)
boxcox_long_r2 <- summary(boxcox_long_model)$r.squared
boxcox_long_r2

# transforming back residuals for comparison
pred <- predict(boxcox_long_model)
res <- long - (pred)^(1/1.1)
mean(res^2) #mse


par(mfrow=c(2,2))
plot(boxcox_long_model)



# PART 3 - 1
# Regression regularized by L2- Ridge regression for lattitude
lat_ridge_model <- cv.glmnet(x_features, lat, alpha = 0)
plot(lat_ridge_model)

# min error and regularized coefficient
min(lat_ridge_model$cvm)
lat_ridge_model$lambda.min
coef(lat_lasso_model, s=lat_ridge_model$lambda.min)


# Regression regularized by L2 - Ridge regression for longitude
long_ridge_model <- cv.glmnet(x_features, long, alpha = 0)
plot(long_ridge_model)

# min error and regularized coefficient
min(long_ridge_model$cvm)
long_ridge_model$lambda.min



# PART 3 - 2
# Regression regularized by L1 - Lasso regression for lattitude
set.seed(42)
lat_lasso_model <- cv.glmnet(x_features,lat, alpha = 1)
plot(lat_lasso_model)

# min error and regularized coefficient
min(lat_lasso_model$cvm)
lat_lasso_model$lambda.min
nnzero(coef(lat_lasso_model, s=lat_lasso_model$lambda.min))
summary(lat_lasso_model$glmnet.fit)

# Regression regularized by L1 - Lasso regression for longitude
set.seed(111)
long_lasso_model <- cv.glmnet(x_features, long, alpha = 1)
plot(long_lasso_model)

# min error and regularized coefficient
min(long_lasso_model$cvm)
long_lasso_model$lambda.min
nnzero(coef(long_lasso_model, s = long_lasso_model$lambda.min))
summary(long_lasso_model$glmnet.fit)



# PART 3 - 3
# Elastic net regression
alphas <- c(.25,.5,.75)
lat_lambdas <- c()
lat_mse <- c()
long_lambdas <- c()
long_mse <- c()

for(a in alphas){
  elnet_lat <- cv.glmnet(x_features, lat, alpha = a)
  elnet_long <- cv.glmnet(x_features,long, alpha = a)
  lat_mse <- c(lat_mse, min(elnet_lat$cvm))
  lat_lambdas <- c(lat_lambdas, elnet_lat$lambda.min)
  long_mse <- c(long_mse, min(elnet_long$cvm))
  long_lambdas <- c(long_lambdas, elnet_long$lambda.min)
  plot(elnet_lat)
  plot(elnet_long)
}


lat_mse
lat_lambdas
min(long_mse)
long_lambdas


# PROBLEM - 2
cc_data <- read_excel("taiwan_cc.xlsx")
index <- createDataPartition(cc_data$`default payment next month`, p=0.8, list=FALSE)
train <- cc_data[index,]
test <- cc_data[-index,]

x_features_train <- as.matrix(train[,seq(2,24)])
y_train <- as.factor(unlist(train[,25]))
x_features_test <- as.matrix(test[,seq(2,24)])
y_test <- as.factor(unlist(test[,25]))
y_test_list <- test[,25]

x <- as.matrix(cc_data[,seq(2,24)])
y <- as.factor(unlist(cc_data[,25]))
  
num_examples <- dim(cc_data)[1]
num_features <- dim(cc_data)[2]


# logistic regression
# using glm cross validation
cc_model <- glm(y ~ x, family = "binomial")
summary(cc_model)

cv.err <- cv.glm(cc_data,cc_model, K=10)$delta[1]
cv.err

# report accuracy
acc_glm <- 1 - cv.err # ~81.4%
acc_glm


#glmnet regression for comparison
#setting up lambda close to 0 to neget the effect of regularization and can be compared with linear model
 glm_model <- cv.glmnet(x_features_train,y_train,alpha =0,family="binomial",type.measure = "class", lambda = c(1e-12, 1e-11))
 glm_model$cvm
 plot(glm_model)
 
 pred_glm <- predict(glm_model, x_features_test,type="class", s="lambda.min")
 correct <- sum(y_test == pred_glm)
 err_rate <- (1-correct/dim(pred_glm))
 acc_glm <- 1 - err_rate
 acc_glm
 ftable(y_test,pred_glm)



par(mfrow=c(2,2))
plot(cc_model)


# fitting regularized models for comparison
glmnet0 <- cv.glmnet(x_features_train,y_train, alpha = 0, family="binomial", type.measure = "class") #ridge
glmnet2 <- cv.glmnet(x_features_train,y_train, alpha = 0.2, family="binomial", type.measure = "class") 
glmnet4 <- cv.glmnet(x_features_train,y_train, alpha = 0.4, family="binomial", type.measure = "class")
glmnet5 <- cv.glmnet(x_features_train,y_train, alpha = 0.5, family="binomial", type.measure = "class") 
glmnet6 <- cv.glmnet(x_features_train,y_train, alpha = 0.6, family="binomial", type.measure = "class") 
glmnet8 <- cv.glmnet(x_features_train,y_train, alpha = 0.8, family="binomial", type.measure = "class") 
glmnet1 <- cv.glmnet(x_features_train,y_train, alpha = 1, family="binomial", type.measure = "class") #lasso

# report lambda value for all regularized models
glmnet0$lambda.min
glmnet2$lambda.min
glmnet4$lambda.min
glmnet5$lambda.min
glmnet6$lambda.min
glmnet8$lambda.min
glmnet1$lambda.min

# variable used in the model
nnzero(coef(glmnet0, s=glmnet0$lambda.min))
nnzero(coef(glmnet2, s=glmnet2$lambda.min))
nnzero(coef(glmnet4, s=glmnet4$lambda.min))
nnzero(coef(glmnet5, s=glmnet5$lambda.min))
nnzero(coef(glmnet6, s=glmnet6$lambda.min))
nnzero(coef(glmnet8, s=glmnet8$lambda.min))
nnzero(coef(glmnet1, s=glmnet1$lambda.min))


# calculate misclassification error for all regularized models
pred0 <- predict(glmnet0, newx = x_features_test, s = "lambda.min", type = "class")
pred2 <- predict(glmnet2, newx = x_features_test, s = "lambda.min", type = "class")
pred4 <- predict(glmnet4, newx = x_features_test, s = "lambda.min", type = "class")
pred5 <- predict(glmnet5, newx = x_features_test, s = "lambda.min", type = "class")
pred6 <- predict(glmnet6, newx = x_features_test, s = "lambda.min", type = "class")
pred8 <- predict(glmnet8, newx = x_features_test, s = "lambda.min", type = "class")
pred1 <- predict(glmnet1, newx = x_features_test, s = "lambda.min", type = "class")

mean(y_test == pred0)
mean(y_test == pred2)
mean(y_test == pred4)
mean(y_test == pred5)
mean(y_test == pred6)
mean(y_test == pred8)
mean(y_test == pred1)

# plot the models
par(mfrow=c(1,1))
plot(glmnet0)
plot(glmnet2)
plot(glmnet4)
plot(glmnet5)
plot(glmnet6)
plot(glmnet8)
plot(glmnet1)


# -----------------------------------------------------------------------
# TEST CODE USED FOR EXERCICE PURPOSE
# -----------------------------------------------------------------------



# glmnet0$glmnet.fit$dev.ratio[which(glmnet0$glmnet.fit$lambda == glmnet0$lambda.min)] 
# glmnet2$glmnet.fit$dev.ratio[which(glmnet2$glmnet.fit$lambda == glmnet2$lambda.min)] 
# glmnet4$glmnet.fit$dev.ratio[which(glmnet4$glmnet.fit$lambda == glmnet4$lambda.min)] 
# glmnet5$glmnet.fit$dev.ratio[which(glmnet5$glmnet.fit$lambda == glmnet5$lambda.min)] 
# glmnet6$glmnet.fit$dev.ratio[which(glmnet6$glmnet.fit$lambda == glmnet6$lambda.min)] 
# glmnet8$glmnet.fit$dev.ratio[which(glmnet8$glmnet.fit$lambda == glmnet8$lambda.min)] 
# glmnet1$glmnet.fit$dev.ratio[which(glmnet1$glmnet.fit$lambda == glmnet1$lambda.min)] 




