# load libraries
library(dplyr)
library(data.table)
library(factoextra)
library(svd)
library(ggrepel)
library(grid)

options(scipen=10000)

# PART 1
# define variables
labels <- read.table("batches.meta.txt")
train.images.rgb <- list()
train.images.lab <- list()
test.images.rgb <- list()
test.images.lab <- list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# custom functions

# function to run sanity check on photos & labels import
drawImage <- function(index) {
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  img <- train.images.rgb[[index]]
  img.r.mat <- matrix(img$r, ncol=32, byrow = TRUE)
  img.g.mat <- matrix(img$g, ncol=32, byrow = TRUE)
  img.b.mat <- matrix(img$b, ncol=32, byrow = TRUE)
  img.col.mat <- rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
  dim(img.col.mat) <- dim(img.r.mat)
  
  # Plot and output label
  library(grid)
  grid.raster(img.col.mat, interpolate=FALSE)
  
  # clean up
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
  
  labels[[1]][train.images.lab[[index]]]
}

#function to plot images
drawImage(sample(1:(num.images*6), size=1))


#mean image drawing
disp_img <- function(img) {
  r <- img[1:1024]
  g <- img[1025:2048]
  b <- img[2049:3072]
  img_matrix = rgb(r,g,b,maxColorValue=255)
  dim(img_matrix) = c(32,32)
  img_matrix = t(img_matrix) # fix to fill by columns
  grid.raster(img_matrix, interpolate=T)
}

# check mean image reconstruction
disp_img(mean_train_images[,1:3072])


# custom function to read the datasets
read.cifar <- function(filenames,num.images = 10000){
  images.rgb <- list()
  images.lab <- list()
  
  # Cycle through all 5 binary files for train data
  for (f in 1:length(filenames)) {
    to.read <- file(paste(filenames[f],sep=""), "rb")
    for(i in 1:num.images) {
      l <- readBin(to.read, integer(), size=1, n=1, endian="big")
      r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
      index <- num.images * (f-1) + i
      images.rgb[[index]] = data.frame(r, g, b)
      images.lab[[index]] = l+1
    }
    close(to.read)
    remove(l,r,g,b,f,i,index, to.read)
  }
  return(list("images.rgb"=images.rgb,"images.lab"=images.lab))
}


#train dataset
cifar_train <- read.cifar(filenames = c("data_batch_1.bin","data_batch_2.bin","data_batch_3.bin",
                                            "data_batch_4.bin","data_batch_5.bin"))
train.images.rgb <- cifar_train$images.rgb
train.images.lab <- cifar_train$images.lab
rm(cifar_train)

#test dataset
cifar_test <- read.cifar(filenames =c("test_batch.bin"))
test.images.rgb <- cifar_test$images.rgb
test.images.lab <- cifar_test$images.lab
rm(cifar_test)


# flatten the data
flat_dt <- function(x_listdata, y_listdata){
  
  #flatten x var
  x_listdata <- lapply(x_listdata,function(x){unlist(x)})
  x_listdata <- do.call(rbind,x_listdata)
  
  #flatten y var
  y_list_data <- lapply(y_listdata,function(x){a=c(rep(0,10)); a[x]=1;return(a)})
  y_listdata <- do.call(rbind,y_listdata)
  
  #return x and y
  return(data.frame("images"=x_listdata,"labels"=y_listdata))
}


#generate flattened train and test datasets
train_dt <- flat_dt(x_listdata=train.images.rgb, y_listdata=train.images.lab)
test_dt <- flat_dt(x_listdata=test.images.rgb,y_listdata =test.images.lab)


# full dataset
cifar_dt <- rbind(train_dt,test_dt)
str(cifar_dt)
summary(cifar_dt)

# calculate the mean image for each category
mean_images <- aggregate(cifar_dt[,1:3072], list(cifar_dt$labels), mean)
str(mean_images)

# create dataset per class
cifar_class <- split(cifar_dt,cifar_dt$labels)
str(cifar_class)

airplane_dt <- cifar_class[[1]]
automobile_dt <- cifar_class[[2]]
bird_dt <- cifar_class[[3]]
cat_dt <- cifar_class[[4]]
deer_dt <- cifar_class[[5]]
dog_dt <- cifar_class[[6]]
frog_dt <- cifar_class[[7]]
horse_dt <- cifar_class[[8]]
ship_dt <- cifar_class[[9]]
truck_dt <- cifar_class[[10]]

# for part 3 calculations - build dataset by category
images.rgb.df.1 <- cifar_class[[1]][,1:3072]
images.rgb.df.2 <- cifar_class[[2]][,1:3072]
images.rgb.df.3 <- cifar_class[[3]][,1:3072]
images.rgb.df.4 <- cifar_class[[4]][,1:3072]
images.rgb.df.5 <- cifar_class[[5]][,1:3072]
images.rgb.df.6 <- cifar_class[[6]][,1:3072]
images.rgb.df.7 <- cifar_class[[7]][,1:3072]
images.rgb.df.8 <- cifar_class[[8]][,1:3072]
images.rgb.df.9 <- cifar_class[[9]][,1:3072]
images.rgb.df.10 <- cifar_class[[10]][,1:3072]


# calculate 20 principal components for each class
airplane_pca <- prcomp(airplane_dt[,1:3072],rank. = 20) 
automobile_pca <- prcomp(automobile_dt[,1:3072],rank. = 20) 
bird_pca <- prcomp(bird_dt[,1:3072],rank. = 20) 
cat_pca <- prcomp(cat_dt[,1:3072],rank. = 20) 
deer_pca <- prcomp(deer_dt[,1:3072],rank. = 20) 
dog_pca <- prcomp(dog_dt[,1:3072],rank. = 20) 
frog_pca <- prcomp(frog_dt[,1:3072],rank. = 20) 
horse_pca <- prcomp(horse_dt[,1:3072],rank. = 20) 
ship_pca <- prcomp(ship_dt[,1:3072],rank. = 20) 
truck_pca <- prcomp(truck_dt[,1:3072],rank. = 20) 


# PCA'S for part 3 calculations
prcomp.label.1 <- airplane_pca
prcomp.label.2 <- automobile_pca
prcomp.label.3 <- bird_pca
prcomp.label.4 <- cat_pca
prcomp.label.5 <- deer_pca
prcomp.label.6 <- dog_pca
prcomp.label.7 <- frog_pca
prcomp.label.8 <- horse_pca
prcomp.label.9 <- ship_pca
prcomp.label.10 <- truck_pca


# create data frame containing class and respestive error for plotting
# errors are sum of al discarded eigenvalues i.e. values from index 21 to 3072 for each class
class <- c("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
error <- c(sum(get_eigenvalue(airplane_pca)[21:3072,1]),sum(get_eigenvalue(automobile_pca)[21:3072,1]),
          sum(get_eigenvalue(bird_pca)[21:3072,1]),sum(get_eigenvalue(cat_pca)[21:3072,1]),
          sum(get_eigenvalue(deer_pca)[21:3072,1]),sum(get_eigenvalue(dog_pca)[21:3072,1]),
          sum(get_eigenvalue(frog_pca)[21:3072,1]),sum(get_eigenvalue(horse_pca)[21:3072,1]),
          sum(get_eigenvalue(ship_pca)[21:3072,1]),sum(get_eigenvalue(truck_pca)[21:3072,1]))
  

# error dataframe to plot
error_df <- data.frame(class,error)

# plot error for all classes
ggplot(error_df, aes(x=class,y=error)) +
  geom_bar(stat = "identity",fill="steelblue", legend=T) + 
  xlab("Class") + 
  ylab("Error") + 
  labs(title = "Error plot for Image Class Vs Error") +
  scale_y_continuous(expand = c(0, 0))


#load pc
# cifar_pca$rotation[,1:20]
# #pc summaries
# summary(cifar_pca)
# #scree plot 
# screeplot(cifar_pca, type = "lines")
# #sqrt of eigenvalues
# sum(airplace_pca$sdev)


# PART 2
# calculate multi dimension scaling for mean image

# create distance matrix for mean images using euclidean distance
mean_image_d_matrix <- dist(mean_images, method = "euclidean")
mean_image_d_matrix

# create multi dimension scaling for distance matrix
mds <- cmdscale(mean_image_d_matrix,k=2 , eig=TRUE)


# display coordinates
mds$points

# calculate the data point distances
x <- mds$points[,1]
y <- mds$points[,2]

# dataframe to plot using distances
mds_df <- data.frame(x,y)
mds_df <- cbind(mds_df,class)

# plot 2D map of distance between each pair of categories
ggplot(mds_df, aes(x=x,y=y))+
  geom_point() +
  labs(title = "2D Map of means for class distances") +
  geom_label_repel(
    aes(x, y, label=class, fill=class), fontface = 'bold', color = 'black', box.padding = 0.35, point.padding = 0.5,
    segment.color = 'grey50', show.legend = FALSE) 


# PART 3

# create placeholder distance matrix
d_mat <- matrix(0L,nrow = 10,ncol = 10)

for(l1 in 1:10){
  img_matrix <- as.matrix(eval(parse(text = paste("images.rgb.df.",l1,sep=""))))
  img_mean_matrix <- matrix(data = as.numeric(mean_images[l1,1:3072]), nrow = 6000, ncol = 3072, byrow = TRUE)
  img_matrix <- img_matrix - img_mean_matrix
  
  for(l2 in 1:10){
    assign(paste("images.rgb.reconst.df.",l1,".",l2,sep = ""),matrix(nrow = 0,ncol = 3072))
    pca_sum <- 0
    
    for(pc in 1:20){
      rprcomp <- eval(parse(text = paste("prcomp.label.",l2,sep="")))$rotation[,pc]
      tprcomp <- t(rprcomp)
      pca_sum = pca_sum + ((img_matrix %*% as.vector(tprcomp)) %*% rprcomp)
      
      assign(paste("images.rgb.reconst.df.",l1,".",l2,sep=""),(img_mean_matrix + pca_sum))
      
      Err = sum(rowSums((eval(parse(text = paste("images.rgb.df.",l1,sep=""))) - eval(parse(text = paste("images.rgb.reconst.df.",l1,".",l2,sep=""))))^2))/6000
      d_mat[l1,l2] <- d_mat[l1,l2] + Err
      d_mat[l2,l1] <- d_mat[l2,l1] + Err
      
      eval(parse(text = paste("rm(images.rgb.reconst.df.",l1,".",l2,")",sep="")))
    }
  }
}


# calculate distance matrix
d_mat <- d_mat / 2
rownames(d_mat) <- class
colnames(d_mat) <- class
class(d_mat)
View(d_mat)


# MDS scaling for distance matrix
mds_d_mat <- cmdscale(d_mat, eig = T, k = 2)
class(mds_d_mat)

# coordinate points
x_d <- mds_d_mat$points[,1]
y_d <- mds_d_mat$points[,2]

mds_d_df <- data.frame(x_d,y_d)
mds_d_df <- cbind(mds_d_df,class)

# plot 2D map of distance between each pair of categories
ggplot(mds_d_df, aes(x=x_d,y=y_d))+
  geom_point() +
  labs(title = "2D map of class similarity distances") +
  geom_label_repel(
    aes(x_d, y_d, label=class, fill=class), fontface = 'bold', color = 'black', box.padding = 0.35, point.padding = 0.5,
    segment.color = 'grey50', show.legend = FALSE) 


plot(x_d,y_d, pch = 16, xlab = "principal components", 
      ylab = "variance explained")
text(x_d,y_d,labels = class)


