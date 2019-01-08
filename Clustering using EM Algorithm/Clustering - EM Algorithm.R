
# loading necessary packages for homework
library(matrixStats)
library(jpeg)
library(ggplot2)
library(gridExtra)

# PROBLEM -1 

# read the data
vocabulary <- unlist(read.csv("vocab.nips.txt", stringsAsFactors = FALSE))
docs <- read.csv("docword.nips.txt", sep = " ", skip = 3, header = FALSE) #remove first three lines from code
colnames(docs) <- c("document","word_id","count")


# create variables used in solution
smoothing_const <- 0.0125
threshold_criteria <- 0.0000001
num_words <- length(docs$word_id) 
topic_count <- 30
document_count <- length(unique(docs[,1]))
vocab_word_count <- 12419



# document vector creation
document_vect <- matrix(0,document_count, vocab_word_count)
for(i in 1:num_words){
  document_vect[docs[i,1], docs[i,2]] = docs[i,3]
}

document_vec <- Matrix::Matrix(document_vec, sparse = TRUE)


# initialization of proabilities and pis based on topic counts
# cluster to 30 topics, using mixture of multinomial topic model
probs <- matrix(0,topic_count,vocab_word_count)
pis_temp <- matrix(1/topic_count,1,topic_count)
str(pis_temp)
#head(probs,10)

# random probabilities assignments such that sum is 1 for each topic
for(i in 1:topic_count){
  rand_vocab_words <- runif(vocab_word_count)
  row_values <- rand_vocab_words/sum(rand_vocab_words)
  probs[i,] <- row_values  #MU'S
}

probs
any(probs==0)

# Q placeholder
term_Qs <- c()

# run EM algorithm till converges
for(k in 1:800){
  # E step 
  # expected value of log likelihood calculations
  # [1500*30] is matrix - sum of features multiplied by probs for each doc and cluster
  inner_u_sum <- document_vect %*% t(log(probs)) 

  # initialize weights placeholder with size [1500*30]
  w_weights <- matrix(0,document_count, topic_count)
  
  #add pi logs
  for(i in seq(topic_count)){
    w_weights[,i] <- inner_u_sum[,i] + log(pis_temp[i])
  }
  
  #calculate wijs
  wijs <- matrix(0, document_count, topic_count)
  ajs <- w_weights
  rowmax <- apply(w_weights, 1, max)
  last_term <- matrix(0,document_count)
  for(i in seq(document_count)){
    last_term[i] <- logSumExp(ajs[i,] - rowmax[i])
  }
  
  # exponentiate w
  w <- ajs - unlist(as.list(rowmax - last_term))
  # normalize weights such that sum is 1
  exponentiated_wijs <- exp(w) 
  for(i in seq(document_count)){
    #calculate wijs - soft weights 
    wijs[i,] <- exponentiated_wijs[i,]/sum(exponentiated_wijs[i,])
  }
  
# final calculations
  final_vals <- w_weights * wijs
  Q <- sum(final_vals)
 #print(Q) 
  term_Qs <- c(term_Qs,Q)

  # M step calculations - updating pis_temp and probs by maximizing it
  for(j in seq(topic_count)){
    
    # updaing p with smoothing
    numerator <- colSums(document_vect * wijs[,j]) + smoothing_const
    denominator <- sum(rowSums(document_vect) * wijs[,j]) + (smoothing_const * vocab_word_count)
    probs[j,] <- numerator/denominator
  
    # update pis_temp
    pis_temp[j] <- sum(wijs[,j]) / document_count
  }

  # stopping criteria to break the looping
  # check convergence using probs
    if(max(abs(probs[k] - probs[k-1])) < threshold_criteria){
      break
    }
}

# plotting pis_temp
barplot(unlist(as.list(pis_temp)), names.arg = seq(1,30,1),main = "Probability Topic is selected",xlab = "Topic",ylab = "Probability")
sum(pis_temp) 


table_to_print <- c()
#10 highest probability words
for(i in seq(topic_count)){
  top10_words <- sort(probs[i,],decreasing = TRUE)[10]
  table_to_print <- c(table_to_print,vocabulary[which(probs[i,] >= top10_words)][1:10])
}

sort(probs[i,],decreasing = TRUE)[10]
vocabulary[which(probs[1,] >= 0.006418717)]

table_to_print <- matrix(table_to_print,nrow = 30)
rownames(table_to_print) <- seq(topic_count)

# show top 10 words with highest probability in each topic
table_to_print <- as.data.frame(table_to_print)
pdf("top_10_words.pdf", height = 11,width=15)
grid.table(table_to_print)
dev.off()
print(table_to_print)



# PROBLEM - 2, PART - 1
# read given test images
img_robert <- readJPEG("RobertMixed.jpg")
img_strelitzia <- readJPEG("smallstrelitzia.jpg")
img_sunset <- readJPEG("smallsunset.jpg")


segment_image <- function(image, segment_size, seed_no){
  
  # prior constant variables based on image
  threshold_criteria <- 0.0125
  number_of_pixels <- prod(dim(image))/3 #based on R,G,B
  img_height <- dim(image)[1]
  img_width <- dim(image)[2]
  
  # get all pixels of the image 
  pixels <- matrix(0,number_of_pixels,3)
  for(i in seq(img_height)){
    for(j in seq(img_width)){
      pixels[((i-1)*img_width)+j,] <- image[i,j,]
    }
  }
  
  
  #scaling pixels 
  pixels <- scale(pixels)
  
  # initializing the cluster center using kmeans
  set.seed(seed_no)
  clust <- kmeans(pixels, center = segment_size, iter.max = 800)
  mus <- as.matrix(clust$centers)
  
  #initialize the pi values 
  pis_temp <- matrix(1/segment_size,1,segment_size)

  #EM steps
  term_Qs <- c()
  for(k in 1:100){ 
    
    # E step
    inner_u_sum <- matrix(0,number_of_pixels, segment_size)
    
    for(i in seq(segment_size)){
      distance_to_mean <- t(t(pixels)-mus[i,]) 
      distance_to_mean <- as.matrix(distance_to_mean)
      inner_u_sum[,i] <- (-1/2) * rowSums(distance_to_mean^2)
    }
    
    # wijs calculations
    wijs_numerator <- exp(inner_u_sum) %*% diag(pis_temp[1:segment_size])
    wijs_denominator <- rowSums(wijs_numerator)
    wijs <- wijs_numerator/wijs_denominator
    
    #print(diag(pis_temp))
    
    # calculate Q
    Q <- sum(inner_u_sum * wijs)
    #print(Q)
    term_Qs <- c(term_Qs,Q)
    
    # M step calculation
    for(j in seq(segment_size)){
      #update mus
      mu_numerator <- colSums(pixels * wijs[,j])
      mu_denominator <- sum(wijs[,j])
      mus[j,] <- mu_numerator/mu_denominator
      
      # update pis_temp 
      pis_temp[j] <- sum(wijs[,j]) / number_of_pixels
    }
    
    # stopping rule
    #check convergence
      if(max(abs(mus[k] - mus[k-1])) < threshold_criteria) {
       break
      }
  }
  
  
  # creating final image
  # need to map pixel to cluster center with highest value of posterior probabilty for that pixel
  final_image <- array(0,c(img_height,img_width,3))
  for(i in seq(img_height)){
    for(j in seq(img_width)){
      idx <- (i-1)*img_width + j
      point <- pixels[idx,]
      mean_seg <- which(wijs[idx,] == max(wijs[idx,]))
      final_image[i,j,] <- mus[mean_seg,]*attr(pixels,'scaled:scale') + attr(pixels,'scaled:center')
    }
  }
  
  imgDm <- dim(final_image)

  imgRGB <- data.frame(
    x = rep(1:imgDm[2], each = imgDm[1]),
    y = rep(imgDm[1]:1, imgDm[2]),
    R = as.vector(final_image[,,1]),
    G = as.vector(final_image[,,2]),
    B = as.vector(final_image[,,3])
  )

  col <- rgb(imgRGB$R,imgRGB$G,imgRGB$B)
  # Plot the image
  ggplot(data = imgRGB, aes(x = x, y = y)) +
    geom_point(colour = col) +
    labs(title = paste("Image with segment size: ", segment_size,  sep="")) +
    xlab("x") +
    ylab("y")
   #dev.off()
}


segment_image(img_robert,10,03252018)
segment_image(img_robert,20,03252018)
segment_image(img_robert,50,03252018)

segment_image(img_strelitzia,10,03252018)
segment_image(img_strelitzia,20,03252018)
segment_image(img_strelitzia,50,03252018)

segment_image(img_sunset,10,03252018)
segment_image(img_sunset,20,03252018)
segment_image(img_sunset,50,03252018)



# PROBLEM - 2, PART - 2
# using one test image to run 5 time with 20 segment_size
# generate image 5 times by setting different seeds each time
# this will generate different initial points to start with
segment_image(img_sunset,20,03252018)
segment_image(img_sunset,20,06012018)
segment_image(img_sunset,20,04152018)
segment_image(img_sunset,20,11052018)
segment_image(img_sunset,20,01222018)


