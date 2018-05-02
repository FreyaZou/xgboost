library(readr)


dtest <- read_csv("~/Dropbox/test.csv")

# install xgboost package, see R-package in root folder
require(xgboost)
require(methods)

testsize <- 550000

dtrain <- read_csv("~/Dropbox/training.csv")
dtrain[33] <- dtrain[33] == "s"
label <- as.numeric(dtrain[[33]])
data <- as.matrix(dtrain[2:31])
weight <- as.numeric(dtrain[[32]]) * testsize / length(label)


sumwpos <- sum(weight * (label==1.0))
sumwneg <- sum(weight * (label==0.0))
print(paste("weight statistics: wpos=", sumwpos, "wneg=", sumwneg, "ratio=", sumwneg / sumwpos))

xgmat <- xgb.DMatrix(data, label = label, weight = weight, missing = -999.0)
param <- list("objective" = "binary:logitraw",
              "scale_pos_weight" = sumwneg / sumwpos,
              "bst:eta" = 0.1,
              "bst:max_depth" = 6,
              "eval_metric" = "auc",
              "eval_metric" = "ams@0.15",
              "silent" = 1,
              "nthread" = 16)
watchlist <- list("train" = xgmat)
nround = 120
print ("loading data end, start to boost trees")
bst = xgb.train(param, xgmat, nround, watchlist );
# save out model
xgb.save(bst, "higgs.model")
print ('finish training')

#############################################
Predict_NoShow_Train <- read_csv("~/Dropbox/Predict_NoShow_Train.csv")

head(Predict_NoShow_Train)

library(dplyr)
#train_sample = sample(Predict_NoShow_Train, nrow(Predict_NoShow_Train) * 0.6)

train_numeric = Predict_NoShow_Train %>%
  select(-Status) %>%
  select(-ID) %>%
  select_if(is.numeric)

str(train_numeric)

day_week = model.matrix(~DayOfTheWeek-1, head(Predict_NoShow_Train))
train_numeric = cbind(train_numeric,day_week) 

Status = ifelse(Predict_NoShow_Train$Status == 'Show-Up',0,1)

### prepare train and test dataset
df = cbind(train_numeric,Status)
str(df)

df1 = apply(df[,-1:-2],2,function(x){as.factor(x)})
df2 = cbind(df[,1:2],df1)
str(df2)
set.seed(415)

data_set_size = floor(nrow(df2)*.6)

indexes = sample(1:nrow(df2), size = data_set_size)

train1 = df2[indexes,]
test1 = df2[-indexes,]


## xgboost
data_set_size = floor(nrow(df2)*.6)

indexes = sample(1:nrow(df2), size = data_set_size)
df3 = apply(df,2,function(x){as.numeric(x)})
train2 = df3[indexes,]
test2 = df3[-indexes,]

train_matrix = data.matrix(train2[,-19])
test_matrix = data.matrix(test2[,-19])

dtrain = xgb.DMatrix(data = train_matrix, label = Status[indexes])
dtest = xgb.DMatrix(data = test_matrix, label = Status[-indexes])

model = xgboost(data=dtrain, max.depth = 18,nround = 10, objective = 'binary:logistic', gamma = 1)
model
pred_xgb = predict(model,dtest)

LogLoss(as.numeric(pred_xgb),as.numeric(test1[,15]))

## logistic regression
mod_logi = glm(Status~.,data = train1, family = 'binomial')
pred.glm = predict(mod_logi,test1[,-15],type = 'response')
LogLoss(as.numeric(pred.glm), as.numeric(test1[,15]))

## random forest
library(mlr)
library(h2o)
traintask <- makeClassifTask(data = data.frame(train_matrix),target = Status) 

library(randomForest)



mod_rf = randomForest(Status~ Age+ DaysUntilAppointment+ Diabetes+ Alcoholism+ Tuberculosis+ Handicapped+ 
                        Smoker+ Scholarship+ Tuberculosis+ RemindedViaSMS+ DayOfTheWeekFriday+ DayOfTheWeekMonday+
                        DayOfTheWeekThursday+ DayOfTheWeekWednesday, 
                      data = train1, importance = T) #, ntree = 10, mtry = 3)
mod_rf
plot(mod_rf)
varImpPlot(mod_rf)

pred1 = predict(mod_rf,test1[,-15],type = 'response')

library(MLmetrics)
LogLoss(as.numeric(pred1), as.numeric(test1[,15]))


obb.err = double(14)
test.err = double(14)

for(mtry in 1:14){
  rf= randomForest(Status~.,data = train1, mtry= mtry, ntree = 400)
  obb.err[mtry] = rf$err.rate[400,1]
  
  pred = predict(rf,test1[,-15])
  test.err[mtry] = with(test1, mean((as.numeric(test1[,15])-pred)^2), as.numeric(test1[,15]))
  
  cat(mtry, " ")
}


matplot(1:14, cbind(obb.err,test.err), pch=19,col=c('red','blue'), type='b', ylab = 'MSE', xlab = '# of predictors considere at each split')
legend('right', legend = c('out off bag error', 'test error'),pch = 19, col=c('red','blue'))


#library('party')
mod_rf2 = cforest(Status~ Age+ DaysUntilAppointment+ Diabetes+ Alcoholism+ Tuberculosis+ Handicapped+ 
                    Smoker+ Scholarship+ Tuberculosis+ RemindedViaSMS+ DayOfTheWeekFriday+ DayOfTheWeekMonday+
                    DayOfTheWeekThursday+ DayOfTheWeekWednesday, 
                  data = train1, 
                  controls = cforest_unbiased(ntree=400, mtry = 2))

#pred2 = predict(mod_rf2,test1[,-15], OOB = TRUE, type = 'response')
# error
#LogLoss(as.numeric(pred2), as.numeric(test1[,15]))

