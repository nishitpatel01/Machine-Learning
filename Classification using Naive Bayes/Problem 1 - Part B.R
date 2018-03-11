
# Problem 1 - Part A
# naive bayes implementation by hand

# reading the dataset
pima_data<-read.csv('pima.csv', header=FALSE)


# split the data into features and associated labels
feature_vector_x <- pima_data[-c(9)]
labels_y <- pima_data[,9]



#replace 0 in features 
x_vector_copy <- feature_vector_x
for (i in c(3, 4, 6, 8)){
  non_values <- feature_vector_x[, i] == 0
  x_vector_copy[non_values, i] = NA
}


# creating placeholders for training/testing results
train_score <- array(dim = 10)
test_score <- array(dim = 10)


# run training/testing 10 times
for (j in 1:10){
  
  # split train and test with 80/20 ratio
  datasplit <- createDataPartition(y = labels_y, p = .8, list = FALSE)
  
  # create training and test feature and labels
  trainx <- x_vector_copy[datasplit,]
  trainy <- labels_y[datasplit]
  
  testx <- x_vector_copy[-datasplit,]
  testy <- labels_y[-datasplit]
  
  
  # divide training data into postive and negative *training* sets
  trposflag <- trainy > 0
  positive_train_examples <- trainx[trposflag, ]
  negative_train_examples <- trainx[!trposflag,]
  
  
  #calculate means and standard deviations
  pos_train_mean <- sapply(positive_train_examples, mean, na.rm = TRUE)
  neg_train_mean <- sapply(negative_train_examples, mean, na.rm = TRUE)
  pos_train_sd <- sapply(positive_train_examples, sd, na.rm = TRUE)
  neg_train_sd <- sapply(negative_train_examples, sd, na.rm = TRUE)
  
  
  # log probability that each feature corresponds to a positive label
  pos_train_offsets <- t(t(trainx) - pos_train_mean)
  pos_train_scaled <- t(t(pos_train_offsets) / pos_train_sd)
  
  pos_train_log_probab <- -(1/2)*rowSums(apply(pos_train_scaled, c(1, 2), function(x)x^2), na.rm=TRUE) - sum(log(pos_train_sd)) + log(nrow(positive_train_examples)/length(trainy))
  
  
  # log probability that each feature vector corresponds to a negative label
  neg_train_offset <- t(t(feature_vector_x) - neg_train_mean)
  neg_train_scaled <- t(t(neg_train_offset) / neg_train_sd)
  neg_train_log_probab <- -(1/2)*rowSums(apply(neg_train_scaled, c(1, 2), function(x)x^2), na.rm=TRUE) - sum(log(neg_train_sd)) + log(nrow(negative_train_examples)/length(trainy))
  
  
  
  # record percentage guessed correctly in scores array
  guess_train_flag <- pos_train_log_probab > neg_train_log_probab
  num_correct_train <- guess_train_flag == trainy
  train_score[j] <- sum(num_correct_train) / (sum(num_correct_train) + sum(!num_correct_train))
  
  
  pos_test_offsets <- t(t(testx) - pos_train_mean)
  pos_test_scaled <- t(t(pos_test_offsets) / pos_train_sd)
  
  # log probability 
  pos_test_log_probab <- -(1/2)*rowSums(apply(pos_test_scaled,c(1, 2), function(x)x^2), na.rm=TRUE) - sum(log(pos_train_sd)) + log(nrow(positive_train_examples)/length(trainy))
  
  
  neg_test_offsets <- t(t(testx) - neg_train_mean)
  neg_test_scaled <- t(t(neg_test_offsets) / neg_train_sd)
  neg_test_log_probab <- -(1/2)*rowSums(apply(neg_test_scaled,c(1, 2), function(x)x^2), na.rm=TRUE) - sum(log(neg_train_sd)) + log(nrow(negative_train_examples)/length(trainy))
  
  
  guess_test_flag <- pos_test_log_probab > neg_test_log_probab
  iscorrect <- guess_test_flag == testy
  test_score[j] <- sum(iscorrect) / (sum(iscorrect)+sum(!iscorrect))
}


#mean of accuracies from cross validation
final_accuracy <- sum(test_score) / length(test_score)
View(final_accuracy)
#74.9%
  
  