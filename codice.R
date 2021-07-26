#Library+import dataset#############
library(ggplot2)
library(randomForest)
library(corrplot)
library(class)
library(neighbr)
library(mclust)
library(factoextra)
library(caret)
library(factoextra)
library(cluster)
library(Gmedian)
library(fpc)
library(dbscan)
library(scales)
library(ggrepel)
library(tidyverse)
library(gridExtra)
library(ggpubr)
library(e1071)
library(caret)
library(NeuralNetTools) 
library(NeuralSens)
library(MLmetrics)
library(keras)
library(dplyr)
library(kfas)
library(lattice)
theme_set(theme_classic()+
            theme(axis.line = element_line(colour = "white"))) 


data <- read.csv("C:/Users/matte/Google Drive (m.baldanza@campus.unimib.it)/Progetto_Machine_Learning/dataset.csv")

#Function#################

norm_feat=function(data){
  sapply(1:ncol(data),
         function(x){
           (data[,x]-min(data[,x]))/(max(data[,x])-min(data[,x]))
         })
}
stand_feat=function(data){
  sapply(1:ncol(data),
         function(x){
           (data[,x]-mean(data[,x]))/sd(data[,x])
         })
}

cake_clust_image=function(clustering_results,cluster_number,data_label){
  val=which(clustering_results==cluster_number)
  table_dat=as.data.frame((table(data_label[val])))
  slices <- table_dat$Freq
  lbls <- as.character(table_dat$Var1)
  pct <- round(slices/sum(slices)*100)
  val=which(pct<10)
  lbls=lbls[-val];pct=pct[-val];slices=slices[-val]
  lbls <- paste(lbls, pct) # add percents to labels
  lbls <- paste(lbls,"%",sep="") # ad % to labels
  pie(slices,labels = lbls, col=rainbow(length(lbls)),
      main=paste("Cluster ->",cluster_number))
}

ggplot_clust_image=function(clustering_results,cluster_number,data_label){
  val=which(clustering_results==cluster_number)
  table_dat=as.data.frame((table(data_label[val])))
  table_dati<- data.frame(group = table_dat$Var1,value = table_dat$Freq)
  low=which(table_dati$value/sum(table_dati$value)<0.10)
  table_dati=table_dati[-low,]
  table_dati$group=factor(table_dati$group)
  table_dati$value=round(table_dati$value/sum(table_dati$value),2)*100
  df2 <- table_dati %>% 
    mutate(csum = rev(cumsum(rev(value))), 
           pos = value/2 + lead(csum, 1),
           pos = if_else(is.na(pos), value/2, pos))
  g1=ggplot(table_dati, aes(x = "" , y = value, fill = fct_inorder(group))) +
    geom_col(width = 1, color = 1) +
    coord_polar(theta = "y") +
    scale_fill_brewer(palette = "Pastel1") +
    geom_label_repel(data = df2,
                     aes(y = pos, label = paste0(value, "%")),
                     size = 4.5, nudge_x = 1, show.legend = FALSE) +
    guides(fill = guide_legend(title = "Group")) +
    theme_void()
  g1
}

correction <- function(sil_result,soglia,n){
  best_pos <- which(sil_result[,3]>soglia)
  best_res <- sil_result[best_pos,]
  best_res[,2] <- n[as.numeric(best_res[,2])]
  return(best_res)
}
sil.fun <- function(data,dist,linkage,n){
  mat.res <- matrix(nrow=length(linkage),ncol=3)
  for (i in 1:length(linkage)){
    dist_clust <- dist(data,method = dist)
    data_hc <- hclust(dist_clust,method = linkage[i])
    vec.sil <- vector(length = length(n))
    for (j in 1:length(n)){
      cluster <- cutree(data_hc,k = n[j])
      sil.1 <- silhouette(cluster,dist = dist_clust)
      vec.sil[j] <- c(mean(sil.1[,3]))
    }
    mat.res[i,] <- c(linkage[i],which.max(vec.sil),as.numeric(max(vec.sil)))  
  }
  return(mat.res)
}

#Pre-Proprocessing-Variables Analysis + Standardization, Normalization#############
etich=data$filename
str(data)
data$chroma_stft
data$label=as.factor(data$label)
colnames(data)
#Delate "filename" feature, irrelevant
data=data[,-1]

sum(is.na(data)) #no missing values

feature=data[,-ncol(data)]

#Check Categorial class, is balance?

table(data$label) #yes
#ten class, classify will be not easy

#Check distribution variables
ggplot(stack(data[,-c(1,length(data))]), aes(x = ind, y = values)) +
  geom_boxplot()+
  theme(axis.text.x=element_text(color = "black", size=8, angle=45, vjust=1, hjust=1))+
  labs(title="Boxplot normalization",x=NULL,y=NULL)


#Check correlation variables
p=cor(data[,-c(1,length(data))])
par(mar=c(2,2,1,1))
corrplot(p)

#Min_Max featuring, range (0,1) Normalization

norm_feature=norm_feat(feature)

norm_feature=as.data.frame(norm_feature)
colnames(norm_feature)=colnames(feature)
ggplot(stack(norm_feature), aes(x = ind, y = values)) +
  geom_boxplot()+
  theme(axis.text.x=element_text(color = "black", size=8, angle=45, vjust=1, hjust=1))+
  labs(title="Boxplot Normalization",x=NULL,y=NULL)

#Standardization
stand_feature=stand_feat(feature)

stand_feature=as.data.frame(stand_feature)
colnames(stand_feature)=colnames(feature)
ggplot(stack(stand_feature), aes(x = ind, y = values)) +
  geom_boxplot()+
  theme(axis.text.x=element_text(color = "black", size=8, angle=45, vjust=1, hjust=1))+
  labs(title="Boxplot Standardization",x=NULL,y=NULL)

# Outlier detenction ---> let's see that mfcc1 have a lot of outlier. This can be a problem if we
# use this variable in our dataset


#Featuring Selection#############

# Featuring selection --> Random Forest methods

p=sqrt(ncol(data))
mod=randomForest(label~., data=data,mtry=p,ntree=2000)
plot(mod)
par(mar=c(2,2,2,1))
varImpPlot(mod,main = "Importance Variables")

#order the importance variable
val=data.frame(DecreaseGini=mod$importance,Names=colnames(feature))
val=val[order(val$MeanDecreaseGini,decreasing = T),]
#take the first ten(values hight)
variable_choice=val$Names[1:10]
variable_choice

new_data=data[,c(variable_choice,"label")]

#Check new boxplot
ggplot(stack(new_data[,-11]), aes(x = ind, y = values)) +
  geom_boxplot()+
  theme(axis.text.x=element_text(color = "black", size=8, angle=45, vjust=1, hjust=1))+
  labs(title="Boxplot normalization",x=NULL,y=NULL)

#we will need a standardization or normalization






















#Investigate Outliers##########
#investigate outlier in mfcc1
values_out=boxplot.stats(new_data$mfcc1)$out
number=c()
for(i in 1:length(values_out)){
  number[i]=which(new_data$mfcc1==values_out[i])
}


#which are the outlier?
mean(new_data$mfcc1[-number])
summary(new_data$mfcc1[-number])
summary(new_data$mfcc1[number])
new_data$mfcc1[number]
new_data$label[number] 
etich[number]


#all the outlier were from classical. Why does it happenen?
#maybe is very hight the distance from the other music, maybe some classical song are very different
#from the rest. What will we do? remove  or maintaine them? I thing that is insane to eliminate
#these observation because they say us much thing


#investigate outlier in mfcc17
values_out=boxplot.stats(new_data$rmse)$out
number=c()
for(i in 1:length(values_out)){
  number[i]=which(new_data$rmse==values_out[i])
}

etich[number] #outlier etichette
#which are the outlier?
new_data$label[number] #a few diferent outlier is not a problem
#Clustering Mixture##########


clustering <- Mclust(new_data[,-11], G=1:9)
par(mar=c(2,2,1,1))
plot(clustering$BIC)
clustering <- Mclust(new_data[,-11], G=4) 

summary(clustering)

classError(clustering$classification, new_data$label) #not important in our analysis

#Graphics

p1=ggplot_clust_image(clustering$classification,1,new_data$label)
p2=ggplot_clust_image(clustering$classification,2,new_data$label)
p3=ggplot_clust_image(clustering$classification,3,new_data$label)
p4=ggplot_clust_image(clustering$classification,4,new_data$label)

plot=ggarrange(p1,p2,p3,p4)
annotate_figure(plot, top = text_grob("Mixture Model-4 Cluster", 
                                      color = "Black", face = "bold", size = 15))

par(mfrow=c(1,1))
#See value
k=as.factor(clustering$classification)
k=length(levels(k))




#K-Means_Iterative Distance##########
norm_selection=norm_feat(new_data[,-11]);colnames(norm_selection)=colnames(new_data[,-11])
stand_selection=stand_feat(new_data[,-11]);colnames(stand_selection)=colnames(new_data[,-11])

df=stand_selection;dz=norm_selection


#K choice

k_cv=5
k_value=vector()
lungh=100
matrice=matrix(nrow=lungh,ncol=10)
colnames(matrice)=c(as.character(c(seq(1,10,1))))
for(j in 1:lungh){
  folds=createFolds(y=new_data$label, k=5, list =T, returnTrain = T)
  matrice_2=matrix(nrow=5,ncol=10)
  colnames(matrice_2)=c(as.character(c(seq(1,10,1))))
  for(i in 1:k_cv){
    train=df[as.vector(folds[[i]]),]
    silh=fviz_nbclust(train, kmeans, method = "silhouette")
    matrice_2[i,]=c(silh$data$y)
  }
  dati=as.data.frame(matrice_2)
  matrice[j,]=apply(dati,2,function(x) mean(x))
}
final_means=apply(dati,2,function(x) mean(x))
final_means
#Best choice is k=3


k3 <- kmeans(df, centers = 3, nstart = 25)
dis = dist(df)^2
sil = silhouette (k3$cluster, dis)

par(mfrow=c(1,1))
par(mar=c(2,1,1,2))
plot(sil)

kmeans <- data.frame(Cluster = k3$cluster, new_data)

par(mfrow=c(1,1))
par(mar=c(1,1,1,1))

p1=ggplot_clust_image(kmeans$Cluster,1,new_data$label)
p2=ggplot_clust_image(kmeans$Cluster,2,new_data$label)
p3=ggplot_clust_image(kmeans$Cluster,3,new_data$label)

plot1=ggarrange(p1,p2,p3)

annotate_figure(plot1, top = text_grob("KMeans-3 Cluster", 
                                       color = "Black", face = "bold", size = 15))



#K-Medoids-Iterative Distance###############

#Choice k

fviz_nbclust(df, pam, method = "wss") #=3

#make thisexample reproducible
set.seed(1)

#perform k-medoids clustering with k = 3 clusters
kmed <- pam(df, k = 3)


dis = dist(df)^2
sil = silhouette (kmed$clustering, dis)
plot(sil)

kmed <- data.frame(Cluster = kmed$clustering, new_data)

p1=ggplot_clust_image(kmed$Cluster,1,new_data$label)
p2=ggplot_clust_image(kmed$Cluster,2,new_data$label)
p3=ggplot_clust_image(kmed$Cluster,3,new_data$label)
plot3=ggarrange(p1,p2,p3)
annotate_figure(plot3, top = text_grob("K-Medoids-3 Cluster", 
                                       color = "Black", face = "bold", size = 15))

#Hierchical Clustering#################
method_dist <- c("euclidean", "maximum", "manhattan", 
                 "canberra", "binary","minkowski")
method_cluster <- c("ward.D", "ward.D2", "single", "complete", 
                    "average", "mcquitty", "median","centroid")

#euclidean distance
#k come tagli pari a 3

data_rf=df
res_euclidean <- sil.fun(data=new_data[,-11],dist = "euclidean",linkage = method_cluster,n=3:10)
res_finale_euclidean <- correction(res_euclidean,soglia=0.50,n=3:10)
res_euclidean
print(res_finale_euclidean)

#best is "average" "0.545941794356864"

hclust_best <- hclust(dist(new_data[,-11]),method = "average")
plot(hclust_best,main="clustering gerarchico")

cut.date_D <- cutree(hclust_best,k = 3)
plot(hclust_best,main="clustering gerarchico")
rect.hclust(hclust_best , k = 3, border = 2:6)


Gruppi <- factor(cut.date_D)
Gruppi
pg_plot <- ggplot(data, aes(x=chroma_stft, y = rmse, col =Gruppi)) +
  geom_point()
pg_plot

ggplot_clust_image(clustering_results = cut.date_D,cluster_number = 1,data_label = data)

g1 <- ggplot_clust_image(clustering_results = cut.date_D,cluster_number = 1,data_label = data$label)
g2 <- ggplot_clust_image(clustering_results = cut.date_D,cluster_number = 2,data_label = data$label)
g3 <- ggplot_clust_image(clustering_results = cut.date_D,cluster_number = 3,data_label = data$label)

plot_set <- ggarrange(g1,g2,g3)
annotate_figure(plot_set, top = text_grob("Clustering gerarchico", 
                                          color = "Black", face = "bold", size = 15))


#manhattan distance
res_man <- sil.fun(data=data_RF,dist = "manhattan",linkage = method_cluster,n=3:10)
res_finale_man <- correction(res_man,soglia=0.50,n=3:10)
res_man
print(res_finale_man)
#best is "ward.D" 0.527912182036855"

hclust_best <- hclust(dist(data_RF),method = "ward.D")
plot(hclust_best)


table(cut.date_D)
hclust_best$labels
table(data$label[which(cut.date_D==1)])
table(data$label[which(cut.date_D==2)])
table(data$label[which(cut.date_D==3)])
ggplot(data_std, aes(x=chroma_stft, y = mfcc4, color = factor(cut.date_D))) +
  geom_point()

#canberra distance
res_man <- sil.fun(data=new_data[,-11],dist = "canberra",linkage = method_cluster,n=3:10)
res_man

#best is "centroid"

hclust_best <- hclust(dist(new_data[,-11]),method = "centroid")
plot(hclust_best)

cut.date_D <- cutree(hclust_best,k = 3)
plot(hclust_best)
rect.hclust(hclust_best , k = 3, border = 2:6)
abline(h = 2400, col = 'red')


ggplot(data, aes(x=chroma_stft, y = mfcc4, color = factor(cut.date_D))) +
  geom_point() 

#binary
res_man <- sil.fun(data=new_data[,-11],dist = "binary",linkage = method_cluster,n=3:10)
res_finale_man <- correction(res_man,soglia=0.50,n=3:10)
res_man
print(res_finale_man)
#best is "centroid"

hclust_best <- hclust(dist(new_data[,-11]),method = "centroid")
plot(hclust_best)

cut.date_D <- cutree(hclust_best,k = 3)
plot(hclust_best)
rect.hclust(hclust_best , k = 3, border = 2:6)
abline(h = 2400, col = 'red')


ggplot(data, aes(x=chroma_stft, y = mfcc4, color = factor(cut.date_D))) +
  geom_point()

#minkowski
res_man <- sil.fun(data=new_data[,-11],dist = "minkowski",linkage = method_cluster,n=3:10)
res_finale_man <- correction(res_man,soglia=0.50,n=3:10)
res_man
print(res_finale_man)
#best is "average" "0.545941794356864" pari a quelli sopra

hclust_best <- hclust(dist(new_data[,-11]),method = "average")
plot(hclust_best)

cut.date_D <- cutree(hclust_best,k = 3)
plot(hclust_best)
rect.hclust(hclust_best , k = 3, border = 2:6)


#Dbscan clustering###################

set.seed(123)

eps=seq(0.01,2,by=0.1)
points=seq(1,40,by=1)
matrice=matrix(ncol=3,nrow=length(eps)*length(points))
colnames(matrice)=c("Eps","Points","Avd_Media")
length(eps)

a=1
pb <- txtProgressBar(0, length(eps), style = 3)
for(i in 1:length(eps)){
  setTxtProgressBar(pb, i)
  for(j in 1:length(points)){
    db = dbscan(dz[,1:8], eps = eps[i], minPts = points[j])
    val=which(db$cluster==0)
    db$cluster=db$cluster[-val]
    if(length(table(db$cluster))>2){
      distance=as.data.frame(dz[,1:8]);distance=distance[-val,]
      silu=silhouette(db$cluster,dist(distance))
      avd_media=mean(silu[,3])
    } else {
      avd_media=0
    }
    matrice[a,]=c(eps[i],points[j],avd_media)
    a=a+1
  }
  Sys.sleep(time = 1)
}
close(pb)

val=which.max(matrice[,3])
matrice[val,]
(matrice[order(matrice[,3],decreasing = T),])

db1 <- dbscan((dz[,1:8]), eps = 0.21, minPts =33)
table(db1$cluster)

silu=silhouette(db1$cluster,dist = dist(dz))
plot(silu)

p1=ggplot_clust_image(db1$cluster,1,new_data$label)
p2=ggplot_clust_image(db1$cluster,2,new_data$label)
p3=ggplot_clust_image(db1$cluster,3,new_data$label)

plot4=ggarrange(p1,p2,p3)
annotate_figure(plot4, top = text_grob("DbScan-3 Cluster", 
                                       color = "Black", face = "bold", size = 15))

#Classification...Dataset Preparation--->Splitting#############
set.seed(10932)

val=createDataPartition(y=new_data$label,times = 1,p = 0.80,list = TRUE)
training_set=new_data[val$Resample1,]
training_set$label=as.factor(training_set$label)
test_set=new_data[-val$Resample1,]
test_set$label=as.factor(test_set$label)


#With KNN Classification#############

#Training Study-->Generalised error
trc=trainControl(method="repeatedcv",repeats = 5,number = 20)

knn_ge=train(label~.,data=training_set,method="knn",trControl=trc,tuneLength = 20,preProcess = c("scale"))

plot(knn_ge)

1-knn_ge$results[4,2] #generalised error

#best choice k=11

#Training set study--->empirical error

mod_knn=class::knn(training_set[,-11],training_set[,-11],cl=training_set$label,k=11,prob=F)
cfmatrix=confusionMatrix(as.factor(training_set$label),mod_knn)
1-(accuracy=cfmatrix$overall[1]) #errors


###Final test error with best model
knnPredict <- predict(knn_ge,newdata=test_set[,-11])
cfM=confusionMatrix(knnPredict,(test_set[,11]))
cfM
cfM$byClass

#With SVM Classification#########

trc=trainControl(method="repeatedcv",repeats = 5,number = 20)

##Radial

# Fit the model 
svm <- train(label ~., data = training_set, method = "svmRadial", trControl = trc, preProcess = c("scale"), tuneGrid = expand.grid(C = seq(1,10,by=0.5),sigma=c(0.1,0.5,1,2)))
# Print the best tuning parameter sigma and C that maximizes model accuracy
svm
svm$bestTune #c=3.5,sigma=0.1

#Best model
svm <- train(label ~., data = training_set, method = "svmRadial", trControl = trc, preProcess = c("scale"), tuneGrid = expand.grid(C = c(3.5),sigma=c(0.1)))
svm

##Polynomial
# Fit the model 
svm1 <- train(label ~., data = training_set, method = "svmPoly", trControl = trc, preProcess = c("scale"), tuneLength = 5)
# Print the best tuning parameter sigma and C that maximizes model accuracy
svm1
svm1$bestTune

#Best is Radial

###training--->empirical error

m <- svm(label~., ,data = training_set,scale = T, kernel="radial", cost=3.5,gamma = 0.1)
prevision=predict(m,training_set[,-11])

cf=confusionMatrix(prevision, training_set[,11]) 
cf

## Test set

previ=predict(svm,test_set[,-11])
ctest=confusionMatrix(previ, as.factor(test_set[,11]))
ctest
ctest$byClass



#With Random Forest Classification#########
#Create control function for training with 10 folds and keep 3 folds for training. search method is grid.
control <- trainControl(method='repeatedcv', 
                        number=20, 
                        repeats=5, 
                        search='grid')
#create tunegrid with 15 values from 1:10 for mtry to tunning model. Our train function will change number of entry variable at each split according to tunegrid. 
tunegrid <- expand.grid(.mtry = (1:10)) 

modellist <- list()
ntree=5000
for (ntree in c(1000, 1500, 2000, 2500)){
  fit <- train(label ~ ., 
               data = training_set,
               method = 'rf',
               metric = 'Accuracy',
               tuneGrid = tunegrid,
               trControl=control,
               ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}



modellist[[1]]$results[which.max(modellist[[1]]$results$Accuracy),] 
modellist[[2]]$results[which.max(modellist[[2]]$results$Accuracy),] 
modellist[[3]]$results[which.max(modellist[[3]]$results$Accuracy),] 
modellist[[4]]$results[which.max(modellist[[4]]$results$Accuracy),] #best 
#best mtry=7 nmin=2500

1-modellist[[4]]$results[which.max(modellist[[4]]$results$Accuracy),"Accuracy"]


#error empiric, stimo su training e valuto su training

rf_emp <- randomForest(label~., data=training_set,mtry=7,ntree=2500)

cfmatrix_rf=confusionMatrix(as.factor(training_set$label),as.factor(rf_emp$predicted))
cfmatrix_rf 


previsioni <- predict(modellist[[4]],test_set)
cfmatrix_rf_test <- confusionMatrix(as.factor(previsioni),as.factor(test_set$label))
cfmatrix_rf_test 

#Reti Neurali########################

## Training-Validation vs Test set 

TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=10)

ctrl  <- trainControl(method='repeatedcv', 
                      number=20, 
                      repeats=10,  
                      summaryFunction = multiClassSummary, 
                      classProbs=T,
                      savePredictions = T) 

set.seed(150) 

TrainingDataIndex <- createDataPartition(data$label, p=0.75, list = FALSE)

data$label=as.factor(data$label)

testData= data[-TrainingDataIndex,]


trainData = data[TrainingDataIndex,]

## Training model with different sizes of Hidden layer and different starting decay 
NNModel <- train(label ~. ,data = trainData,
                 method = "nnet",
                 preProcess = c("scale"),
                 trControl= ctrl,
                 tuneGrid=expand.grid(size=c(seq(5,25,1)),decay=seq(0,0.5,0.01)))

NNModel

res = NNModel$results

## finding the copuple (decay, size) best in terms of accuracy of valodation 
which.max(res[,3])

## Neural Network with the best (size,decay)

fit.mlp <- train(label ~ ., data = trainData, 
                 method = "nnet",
                 trControl = ctrl, 
                 preProcess=c("scale"),
                 maxit = 800, # Maximum number of iterations
                 tuneGrid = expand.grid(size=c(10),decay=c(0.44)),
                 metric = "Accuracy")
#Generalized Error 
fit.mlp$results$Accuracy
fit.mlp
Gen_Err = (1-fit.mlp$results$Accuracy)

#Empirical Error
Pred_train = predict(fit.mlp, trainData)
cmNN_train = confusionMatrix(Pred_train, trainData[,27])
cmNN_train$overall[1]
Emp_Err = (1-cmNN_train$overall[1])

# Create confusion matrix

NNPredictions.mlp = predict(fit.mlp, testData[,-27])
cmNN <-confusionMatrix(NNPredictions.mlp, testData$label)
cmNN$overall #Accuracy
cmNN$table
cmNN$byClass

Test_err= 1-cmNN$overall[1]

# Great performance!

# Graphic

cmNN_ref = confusionMatrix(factor(NNPredictions.mlp), factor(testData$label), dnn = c("Prediction", "Reference"))
cmNN_ref

plt = as.data.frame(cmNN_ref$table)

plt$Prediction
#test due

plt$Prediction = factor(plt$Prediction, levels=rev(levels(plt$Prediction)))

ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c("rock","reggae","pop","metal","jazz","hiphop","disco","country","classical","blues")) +
  scale_y_discrete(labels=c("blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"))
