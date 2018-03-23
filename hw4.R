
# HW4
# PROBLEM 1 - PART 1

# load necessary packages 
library(cluster)    
library(factoextra) 
library(dendextend) 
library(ape)
library(purrr)
library(gridExtra)
library(caret)
library(reshape2)
library(randomForest)
library(data.table)
library(dplyr)
library(pracma)
library(readxl)


# load dataset
euro_data <- read.csv("european_data.csv")

# convert country column into row name
euro_data <- data.frame(euro_data[,-1],row.names = euro_data[,1])

# scaling
euro_data <- scale(euro_data)
head(euro_data)


# dissimilarity matrix
euro_data_d <- dist(euro_data, method = "euclidean")

# cluseting using single, complete and average link
euro_clst_single <- hclust(euro_data_d, method = "single")
euro_clst_complete <- hclust(euro_data_d, method = "complete")
euro_clst_average <- hclust(euro_data_d, method = "average")

# plot dendograms
plot(euro_clst_single, cex=0.8, hang= -1,  main = "Cluster Dendogram - Single link")
plot(euro_clst_complete, cex=0.8, hang= -1, main = "Cluster Dendogram - Complete link")
plot(euro_clst_average, cex=0.8, hang= -1, main = "Cluster Dendogram - Average link")


# plot using phylogenetic trees
plot(as.phylo(euro_clst_single), type = 'fan', main ="Phylogenetic Tree - Single link")
plot(as.phylo(euro_clst_complete), type = 'fan', main = "Phylogenetic Tree - Complete link")
plot(as.phylo(euro_clst_average), type = 'fan', main = "Phylogenetic Tree - Average link")


# PROBLEM 1 - PART 2

# determine number of clusters
set.seed(1500)

# function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(euro_data[,2:9], k, nstart = 100 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:20

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 20, cex=2,
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method")


# fit kmeans with different cluster sizes for observation
km2 <- kmeans(euro_data, centers=2, nstart = 100)
km3 <- kmeans(euro_data, centers=3, nstart = 100)
km4 <- kmeans(euro_data, centers=4, nstart = 100)
km5 <- kmeans(euro_data, centers=5, nstart = 100)


# plot clusters
clusplot(euro_data, km$cluster, main='2D representation of the Cluster solution',
         color=TRUE, shade=TRUE,
         labels=2, lines=0)

#plot all clusters for comparison
p2 <- fviz_cluster(km2,data=euro_data) + ggtitle("Cluster size = 2")
p3 <- fviz_cluster(km3,data=euro_data) + ggtitle("Cluster size = 3") 
p4 <- fviz_cluster(km4,data=euro_data) + ggtitle("Cluster size = 4")
p5 <- fviz_cluster(km5,data=euro_data) + ggtitle("Cluster size = 5")

grid.arrange(p2,p3,p4,p5,nrow = 2)


# part 3

# custom function to read files per category
read_files <- function(filepath){
  list_of_files <- list.files(path = filepath, recursive = TRUE,
                              pattern = "\\.txt$", full.names = TRUE)
  dt <- rbindlist(sapply(list_of_files, fread, simplify = FALSE), use.names = TRUE, idcol = "FileName" )
}


# read data and create dataframes (put the file in train anf test folder manually and read it separately)
#train data
brush_teeth_tr <- read_files("Brush_teeth")
climb_stairs_tr <- read_files("Climb_stairs")
comb_hair_tr <- read_files("Comb_hair")
descend_stairs_tr <- read_files("Descend_stairs")
drink_glass_tr <- read_files("Drink_glass")
eat_meat_tr <- read_files("Eat_meat")
eat_soup_tr <- read_files("Eat_soup")
getup_bed_tr <- read_files("Getup_bed")
liedown_bed_tr <- read_files("Liedown_bed")
pour_water_tr <- read_files("Pour_water")
sitdown_chair_tr <- read_files("Sitdown_chair")
standup_chair_tr <- read_files("Standup_chair")
use_telephone_tr <- read_files("Use_telephone")
Walk_tr <- read_files("Walk")


#test data
brush_teeth_te <- read_files("Brush_teeth")
climb_stairs_te <- read_files("Climb_stairs")
comb_hair_te <- read_files("Comb_hair")
descend_stairs_te <- read_files("Descend_stairs")
drink_glass_te <- read_files("Drink_glass")
eat_meat_te <- read_files("Eat_meat")
eat_soup_te <- read_files("Eat_soup")
getup_bed_te <- read_files("Getup_bed")
liedown_bed_te <- read_files("Liedown_bed")
pour_water_te <- read_files("Pour_water")
sitdown_chair_te <- read_files("Sitdown_chair")
standup_chair_te <- read_files("Standup_chair")
use_telephone_te <- read_files("Use_telephone")
Walk_te <- read_files("Walk")


# function to cut the signal into pieces and the flatten it out into one row per signal grouped by file/signal
final_df <- list()
d_df <- list()
seg_size <- 8
flatten_dt_size <- (1:(seg_size*3))

cut_signal <- function(df,class, segment_size){
  
  splitted_df <- split(df,df$FileName)
  
  for(i in 1:length(splitted_df)){
    chunk <- segment_size
    n <- nrow(splitted_df[[i]])
    r  <- rep(1:floor(n/chunk),each=chunk)[1:n]
    d <- split(splitted_df[[i]],r)
    
    for(j in 1:length(d)){
      dframe <- d[[j]]
      dframe <- do.call(rbind,dframe[,2:4])
      dframe <- do.call(cbind,as.list(dframe))
      dframe <- as.data.frame(dframe)
      dframe$File <- splitted_df[[i]]$FileName[1]
      dframe$class <- class
      d_df[[j]] <- dframe
    }
    final_df[[i]] <- do.call("rbind",d_df)
  }
  as.data.frame(do.call("rbind",final_df))
}


# build features using signal segments 
#train
brush_seg_tr <- cut_signal(brush_teeth_tr,"brush_teeth",seg_size)
climb_stairs_seg_tr <- cut_signal(climb_stairs_tr,"climb_stairs",seg_size)
comb_hair_seg_tr <- cut_signal(comb_hair_tr,"comb_hair",seg_size)
descend_stairs_seg_tr <- cut_signal(descend_stairs_tr,"descend_stairs",seg_size)
drink_glass_seg_tr <- cut_signal(drink_glass_tr,"drink_glass",seg_size)
eat_meat_seg_tr <- cut_signal(eat_meat_tr,"eas_meat",seg_size)
eat_soup_seg_tr <- cut_signal(eat_soup_tr,"eat_soup",seg_size)
getup_bed_seg_tr <- cut_signal(getup_bed_tr,"getup_bed",seg_size)
liedown_bed_seg_tr <- cut_signal(liedown_bed_tr,"liedown_bed",seg_size)
pour_water_seg_tr <- cut_signal(pour_water_tr,"pour_water",seg_size)
sitdown_chair_seg_tr <- cut_signal(sitdown_chair_tr,"sitdown_chair",seg_size)
standup_chair_seg_tr <- cut_signal(standup_chair_tr,"standup_chair",seg_size)
use_telephone_seg_tr <- cut_signal(use_telephone_tr,"use_telephone",seg_size)
walk_seg_tr <- cut_signal(Walk_tr,"walk",seg_size)

#test
brush_seg_te <- cut_signal(brush_teeth_te,"brush_teeth",seg_size)
climb_stairs_seg_te <- cut_signal(climb_stairs_te,"climb_stairs",seg_size)
comb_hair_seg_te <- cut_signal(comb_hair_te,"comb_hair",seg_size)
descend_stairs_seg_te <- cut_signal(descend_stairs_te,"descend_stairs",seg_size)
drink_glass_seg_te <- cut_signal(drink_glass_te,"drink_glass",seg_size)
eat_meat_seg_te <- cut_signal(eat_meat_te,"eas_meat",seg_size)
eat_soup_seg_te <- cut_signal(eat_soup_te,"eat_soup",seg_size)
getup_bed_seg_te <- cut_signal(getup_bed_te,"getup_bed",seg_size)
liedown_bed_seg_te <- cut_signal(liedown_bed_te,"liedown_bed",seg_size)
pour_water_seg_te <- cut_signal(pour_water_te,"pour_water",seg_size)
sitdown_chair_seg_te <- cut_signal(sitdown_chair_te,"sitdown_chair",seg_size)
standup_chair_seg_te <- cut_signal(standup_chair_te,"standup_chair",seg_size)
use_telephone_seg_te <- cut_signal(use_telephone_te,"use_telephone",seg_size)
walk_seg_te <- cut_signal(Walk_te,"walk",seg_size)


# combineall train and test segments
total_tr <- bind_rows(brush_seg_tr,climb_stairs_seg_tr,comb_hair_seg_tr,descend_stairs_seg_tr,
                      drink_glass_seg_tr,eat_meat_seg_tr,eat_soup_seg_tr,getup_bed_seg_tr,liedown_bed_seg_tr,
                      pour_water_seg_tr,sitdown_chair_seg_tr,standup_chair_seg_tr,use_telephone_seg_tr,walk_seg_tr)

total_te <- bind_rows(brush_seg_te,climb_stairs_seg_te,comb_hair_seg_te,descend_stairs_seg_te,
                      drink_glass_seg_te,eat_meat_seg_te,eat_soup_seg_te,getup_bed_seg_te,liedown_bed_seg_te,
                      pour_water_seg_te,sitdown_chair_seg_te,standup_chair_seg_te,use_telephone_seg_te,walk_seg_te)

# remove dups
total_tr <- total_tr[!duplicated(total_tr),]
total_te <- total_te[!duplicated(total_te),]

#update training and test segments after removing dups
splitted_total_tr <- split(total_tr,total_tr$class)
splitted_total_te <- split(total_te,total_te$class)

# Assign train and test features back after removing dups
# train
brush_seg_tr <- splitted_total_tr[[1]]
climb_stairs_seg_tr <-  splitted_total_tr[[2]]
comb_hair_seg_tr <-  splitted_total_tr[[3]]
descend_stairs_seg_tr <-  splitted_total_tr[[4]]
drink_glass_seg_tr <-  splitted_total_tr[[5]]
eat_meat_seg_tr <-  splitted_total_tr[[6]]
eat_soup_seg_tr <-  splitted_total_tr[[7]]
getup_bed_seg_tr <-  splitted_total_tr[[8]]
liedown_bed_seg_tr <-  splitted_total_tr[[9]]
pour_water_seg_tr <-  splitted_total_tr[[10]]
sitdown_chair_seg_tr <-  splitted_total_tr[[11]]
standup_chair_seg_tr <-  splitted_total_tr[[12]]
use_telephone_seg_tr <-  splitted_total_tr[[13]]
walk_seg_tr <-  splitted_total_tr[[14]]

# test
brush_seg_te <- splitted_total_te[[1]]
climb_stairs_seg_te <-  splitted_total_te[[2]]
comb_hair_seg_te <-  splitted_total_te[[3]]
descend_stairs_seg_te <-  splitted_total_te[[4]]
drink_glass_seg_te <-  splitted_total_te[[5]]
eat_meat_seg_te <-  splitted_total_te[[6]]
eat_soup_seg_te <-  splitted_total_te[[7]]
getup_bed_seg_te <-  splitted_total_te[[8]]
liedown_bed_seg_te <-  splitted_total_te[[9]]
pour_water_seg_te <-  splitted_total_te[[10]]
sitdown_chair_seg_te <-  splitted_total_te[[11]]
standup_chair_seg_te <-  splitted_total_te[[12]]
use_telephone_seg_te <-  splitted_total_te[[13]]
walk_seg_te <-  splitted_total_te[[14]]


# build cluster using train
set.seed(1500)

# cluster all train segments using kmeans (kept the clusters size based on highest accuracy)
options(warn=1)
km_sig <- kmeans(as.matrix(total_tr[,1:24]), centers = 480, nstart = 50, iter.max = 30)  #max iteration increased to gain convergence


# function to calculate closest cluster for segment in each class
csize <- c(1:480)
d_frm <- list()
compute_seg_cluster <- function(dataframe,clust_size){
  
  splitted_df <- split(dataframe,dataframe$File)
  l <- length(splitted_df)
  
  for(i in 1:l){
    dist <- pdist2(as.matrix(km_sig$centers), as.matrix(splitted_df[[i]][,1:24]))
    min_index <- apply(dist, 2, which.min)
    dfrm <- histc(min_index,clust_size)$cnt
    dfrm <- do.call(cbind,as.list(dfrm))
    dfrm <- as.data.frame(dfrm)
    dfrm$class <- splitted_df[[i]]$class[1]
    d_frm[[i]] <- dfrm
  }
  final_vec <- do.call("rbind",d_frm)
}


#calculating the cluster center for every signal
brush <- compute_seg_cluster(brush_seg_tr,csize)
clmb_str <- compute_seg_cluster(climb_stairs_seg_tr,csize)
cmb_hr <- compute_seg_cluster(comb_hair_seg_tr,csize)
desd_str <- compute_seg_cluster(descend_stairs_seg_tr,csize)
drk_gls <- compute_seg_cluster(drink_glass_seg_tr,csize)
et_mt <- compute_seg_cluster(eat_meat_seg_tr,csize)
et_sp <- compute_seg_cluster(eat_soup_seg_tr,csize)
gtp_bd <- compute_seg_cluster(getup_bed_seg_tr,csize)
liedn_bd <- compute_seg_cluster(liedown_bed_seg_tr,csize)
pr_et <- compute_seg_cluster(pour_water_seg_tr,csize)
stdn_chr <- compute_seg_cluster(sitdown_chair_seg_tr,csize)
std_chr <- compute_seg_cluster(standup_chair_seg_tr,csize)
use_ph <- compute_seg_cluster(use_telephone_seg_tr,csize)
wk <- compute_seg_cluster(walk_seg_tr,csize)

#calculating the cluster center for every test signal 
brush_e <- compute_seg_cluster(brush_seg_te,csize)
clmb_str_e <- compute_seg_cluster(climb_stairs_seg_te,csize)
cmb_hr_e <- compute_seg_cluster(comb_hair_seg_te,csize)
desd_str_e <- compute_seg_cluster(descend_stairs_seg_te,csize)
drk_gls_e <- compute_seg_cluster(drink_glass_seg_te,csize)
et_mt_e <- compute_seg_cluster(eat_meat_seg_te,csize)
et_sp_e <- compute_seg_cluster(eat_soup_seg_te,csize)
gtp_bd_e <- compute_seg_cluster(getup_bed_seg_te,csize)
liedn_bd_e <- compute_seg_cluster(liedown_bed_seg_te,csize)
pr_et_e <- compute_seg_cluster(pour_water_seg_te,csize)
stdn_chr_e <- compute_seg_cluster(sitdown_chair_seg_te,csize)
std_chr_e <- compute_seg_cluster(standup_chair_seg_te,csize)
use_ph_e <- compute_seg_cluster(use_telephone_seg_te,csize)
wk_e <- compute_seg_cluster(walk_seg_te,csize)


# combine all train and test signals 
train_dt <- bind_rows(brush,clmb_str,cmb_hr,desd_str,drk_gls,et_mt,et_sp,gtp_bd,liedn_bd,pr_et,stdn_chr,std_chr,use_ph,wk)
test_dt <- bind_rows(brush_e,clmb_str_e,cmb_hr_e,desd_str_e,drk_gls_e,et_mt_e,et_sp_e,gtp_bd_e,liedn_bd_e,pr_et_e
                     ,stdn_chr_e,std_chr_e,use_ph_e,wk_e)


# training random forest model on train signals 
train_dt$class <- as.factor(train_dt$class)
model_rf <- randomForest(class ~ ., data = train_dt)
plot(model_rf)
model_rf

pred <- predict(model_rf, newdata=test_dt)
t <- table(pred,test_dt$class)
cm <- confusionMatrix(pred, test_dt$class)
cm_t <- cm$table
cm_t <- as.data.frame.matrix(cm_t)
cm_t <- as.data.frame(cm_t)

View(cm_t)
# current best accuracy (cluster-480, segment-8): 77.3%


# Problem 2 - Part 2
# below observation was taken by individual run of program by changing cluster and segment sizes

accuracies <- read_excel("accuracy vs segments.xlsx", sheet = 2)
accuracies <- as.data.frame(accuracies)


ggplot(accuracies, aes(x=clust,y=accu, group=as.factor(seg),colour=as.factor(seg))) +
  geom_line(size=1) + 
  ylab(label="Accuracy") + 
  xlab("Cluster Size") +
  xlim(100, 480) +
  ylim(60,80) + 
  labs(title = "Plot of Test Accuracy Vs Cluster Size for All segment sizes") 



