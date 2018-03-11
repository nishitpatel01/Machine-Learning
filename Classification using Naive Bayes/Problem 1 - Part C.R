# Problem 1 - Part C
# naive bayes classifier using caret and klar packages

#loading packages
library(caret)
library(klaR)
library(MASS)

data <- read.csv('pima.csv', header = FALSE)

#data split into feature and label
X <- data[,-c(9)]
label_y <- as.factor(data[,9])

#get indicies to use for training
datasplit <- createDataPartition(y = label_y, p = .8, list = FALSE)

#extract train features and labels
X_train <- X[datasplit, ]
y_train <- label_y[datasplit]


# extract testing feature and labels
X_test <- X[-datasplit, ]
y_test <- label_y[-datasplit]


# train the model with cross validation
model_nb <- train(X_train, y_train, 'nb', trControl = trainControl(method = 'cv', number = 10))


# make prediction using test set
pred <- predict(model_nb, newdata = X_test)

# print confusion maxtrix 
confusionMatrix(data = pred, y_test)


#  Reference
#  Prediction  0  1
#           0 87 32
#           1 13 21

# Accuracy : 0.7059     

