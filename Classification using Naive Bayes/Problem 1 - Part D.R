
# Problem 1 - Part D
# using svm light

#read the data
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

# train the model with a SVM
svm <- svmlight(X_train, y_train, pathsvm = 'C:\\Users\\NishitP\\Desktop\\svm_light_windows64\\')

# make prediction using test set
pred <- predict(svm, newdata = X_test)


# compute percent correct classifications
guess<-pred$class


#accuracy
correct <- sum(guess == y_test)
incorrect <- sum(!(guess == y_test))
accuracy <- correct / (correct + incorrect)
accuracy
#83.6%