
#loading necessary package
library(caret)
library(ggplot2)

#reading data
adult_dt <- read.csv('adult_data.csv', header = FALSE)

# check missing values in dataset
sapply(adult_dt, function(x) sum(is.na(x)))

# check & remove records with missing value 
# continuous variables does not have missing values
# adult_dt <- adult_dt[!(adult_dt$V2 == " ?" | adult_dt$V7 == " ?" | adult_dt$V14 ==" ?"),]

# reconstruct the dataset by removing categorical variables and only keeping numerical features
x_vec <- adult_dt[,c(1,3,5,11,12,13)]
y_lab <- adult_dt[,15]

# perform scaling
for (i in 1:6){
  x_vec[i] <- scale(as.numeric(as.matrix(x_vec[i])))
}


# define algorithm parameters
lambdas <- c(0.0001,0.001, 0.01, 0.1, 1)
epochs <- 50
steps <- 300
training_per <- .8
validation_per <- .1
test_per <- .1
num_examples_epoch_test <- 50
steps_til_eval <- 30
steplength_a <- .01
steplength_b <- 50


# get positive and negative examples
# dataset contains 4 classes with a . (dot) and extra space in the begining in it 
# therefore it was cleaned in dataset already by replacing
# " >50k." with ">50k" and " <=50k" with "<50k"
class(adult_dt$V15)
positive <- ">50k" 
negative <- "<=50k" 


#train, test split
datasplit <- createDataPartition(y = y_lab, p = .8, list = FALSE)
train_x <- x_vec[datasplit,]
train_y <- y_lab[datasplit]
other_x <- x_vec[-datasplit,]
other_y <- y_lab[-datasplit]

datasplit2 <- createDataPartition(y = other_y, p = .5, list = FALSE)
testx <- other_x[datasplit2,]
testy <- other_y[datasplit2]
valx <- other_x[-datasplit2,]
valy <- other_y[-datasplit2]


#SVM implementation

# define hinge loss
hinge_loss <- function(predicted, actual){
  return (max(0, 1 - (predicted * actual) ))
}


#evaluation for specific example x (6 items in vector) with parameters a and b
evaluate <- function(x, a, b){
  new_x <- as.numeric(as.matrix(x))
  return (t(a) %*% new_x + b) 
}


# USE ORIGINAL LABEL VALUES FROM DATA FILE AND THEN COMPARE AND UPDATE TO -1 AND 1
#Change y in dataset from <=50k and >50k to -1 and 1
converty <- function(y){
  if(y == negative){
    return (-1)
  }
  else if(y == positive){
    return (1)
  }
  else{
    return(NA)
  }
}

# convert predictions
convert_pred <- function(val){
  if(val >= 0){
    return(1)
  }
  else{
    return(-1)
  }
}


# calculating accuracy
accuracy <- function(x,y,a,b){
  correct <- 0
  wrong <- 0
  for (i in 1:length(y)){
    pred <- evaluate(x[i,], a, b)
    pred <- convert_pred(pred)
    actual <- converty(y[i])
    
    if(pred == actual){
      correct <- correct + 1 
    } else{
      wrong <- wrong + 1
    }
  }
  return(c((correct / (correct + wrong)), correct, wrong))
}


#array of validation accuracies
val_accuracies <- c()
test_accuracies <- c()


final_df <- data.frame(lam = numeric(), accu = double())
mag_df <- data.frame(lam = character(),mag = double())
View(final_df)
View(mag_df)


for (lambda in lambdas){
  
  # initialize a and b vectors
  # a would be same dimension as feature vector
  a <- c(0,0,0,0,0,0)
  b <- 0
  
  # place holders for accuracy and direction
  accuracies <- c()
  mag_a <- c()
  posup <- 0
  negup <- 0
  
  
  for (epoch in 1:epochs){
    
    #setting aside 50 random samples for testing after every 30 steps
    ran_vals <- sample(1:dim(train_x)[1], 50)
    accuracy_data <- train_x[ran_vals, ]
    accuracy_labels <- train_y[ran_vals]
    train_data <- train_x[-ran_vals,]
    train_labels <- train_y[-ran_vals]
    
    #Keep track of the number of steps taken at each epoch for debugging purposes
    num_steps <- 0
    
    for (step in 1:steps){
      
      if(num_steps %% steps_til_eval == 0){
        calc <- accuracy(accuracy_data, accuracy_labels, a, b)
        accuracies <- c(accuracies, calc[1]) 
        
        # calculate magnitude of weight vector a
        mag <- t(a)%*%a #a%*%a
        mag_a <- c(mag_a, mag[1])
      }
      
      k <- sample(1:length(train_labels), 1)
      while(is.na( converty( train_labels[k]))){
        k <- sample(1:length(train_labels), 1)
      }
      
      xex <- as.numeric(as.matrix( train_data[k,] ))
      yex <- converty( train_labels[k] )
      
      pred <- evaluate(xex, a, b)
      steplength = 1 / ((steplength_a * epoch) + steplength_b)
      
      
      # calculate gradient vectors
      if(yex * pred >= 1){
        p1 <- lambda * a
        p2 <- 0
        posup <- posup + 1
      } else {
        p1 <- (lambda * a) - (yex * xex)
        p2 <- -(yex)
        negup <- negup + 1
      }
      
      # update values for a and b by gradient descent
      a <- a - (steplength * p1)
      b <- b - (steplength * p2)
      
      # update steps count
      num_steps <- num_steps + 1
    }
  }
  
  # calculate validation accuracies
  valeval <- accuracy (valx, valy, a, b)
  val_accuracies <- c(val_accuracies, valeval[1])
  
  # calculate test accuracies
  testeval <- accuracy(testx, testy, a, b)
  test_accuracies <- c(test_accuracies, testeval[1])
  
  # creating dataframe to store results for accuracy plot
  for(i in 1:length(accuracies)){
    #print("in this now"+toString(accuracies[i]))
    ldf<-data.frame(lambda,accuracies[i])
    names(ldf)<-c("lambda","acc")
    final_df <- rbind(final_df, ldf)
  }
  
  # creating dataframe to store resutls for weight vector plot
  for (i in 1:length(mag_a)){
    ma <- data.frame(lambda,mag_a[i])
    names(ma) <- c("lambda","mag")
    mag_df <- rbind(mag_df,ma)
  }
}


# add count column to both dataframes
final_df["epoch"] <- rep(1:500,5)
mag_df["epoch"] <- rep(1:500,5)

View(final_df)
View(mag_df)



# part A - plot accuracy curves
ggplot(data = final_df, aes(x = epoch, y = acc)) +
  geom_line(aes(colour=factor(lambda)), size=0.7) +
  xlab("Steps") + 
  ylab("Test Accuracy") + 
  labs(title = "Plot of Test Accuracy Vs Steps") + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0))

# part B - plot magnitude of coefficient vector
ggplot(data = mag_df, aes(x = epoch, y = mag)) + 
  geom_line(aes(colour=factor(lambda)), size=0.7) +
  xlab("Steps") + 
  ylab("Magnitude of weight vector a") + 
  ggtitle("Plot of weight vector magnitude Vs Steps") + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0))

#part 3 - estimate of best value of regularization constant
max_index <- 1
for(i in 1:length(val_accuracies)){
  if (val_accuracies[i] >= val_accuracies[max_index]){
    max_index <- i
  }
}
max_lambda <- lambdas[max_index]
max_lambda

# part 4 - estimate of accuracy of best classifier 
test_accuracies[max_index]



