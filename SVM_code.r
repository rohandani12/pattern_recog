rm(list=ls())
require(e1071)

#--------------------Load data with 2 classes---------------------

train <- read.table("C:\\Users\\Rohan\\Documents\\MATLAB\\train_sp2017_v19.txt",header = FALSE)
test <- read.table("C:\\Users\\Rohan\\Documents\\MATLAB\\test_sp2017_v19.txt",header = FALSE)
train$class <- c(rep(1,5000),rep(2,5000),rep(3,5000))
train <- train[1:10000,]

answer <- as.data.frame(rep(x=c(3,1,2,3,2,1),15000/6))
test$class <- rep(x=c(3,1,2,3,2,1),15000/6)

test<-test[test$class != 3,]
test$class <- as.factor(test$class)
train$class <- as.factor(train$class)

#------------------SVM linear kernel-----------------------

c1 <- subset(train, select=-class)
c2 <- train$class
svmfitting_linear<-svm(c1, c2, kernel='linear')

#--------------Prediction on new data--------------------

pred_data_linear <- predict(svmfitting_linear, test[,1:4])
caret::confusionMatrix(pred_data_linear, test$class) #---print 'svmfitting_linear' in console for accuracy
w_l <- t(svmfitting_linear$coefs) %*% svmfitting_linear$SV

#--------------SVM radial kernel------------------

c1 <- subset(train, select=-class)
c2 <- train$class
svmfitting_radial<-svm(c1, c2, kernel='radial')

#--------------Prediction on new data-----------------

pred_data_radial <- predict(svmfitting_radial,test[,1:4])

caret::confusionMatrix(pred_data_radial, test$class) #---print 'svmfitting_radial' in console for accuracy


w_r <- t(svmfitting_radial$coefs) %*% svmfitting_radial$SV
