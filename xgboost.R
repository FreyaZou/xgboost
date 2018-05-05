Predict_NoShow_Train <- read_csv("~/Dropbox/Predict_NoShow_Train.csv")

df = Predict_NoShow_Train

df$Gender_n = ifelse(df$Gender=='M',1,0)

df$DateAppointmentWasMade = as.Date(df$DateAppointmentWasMade,'%Y-%m-%d')
df$DateOfAppointment = as.Date(df$DateOfAppointment,'%Y-%m-%d')

df$year_of_app_made = as.numeric(format(df$DateAppointmentWasMade, "%Y"))
df$month_of_app_made = as.numeric(format(df$DateAppointmentWasMade, "%m"))

df$year_of_app = as.numeric(format(df$DateOfAppointment, "%Y"))
df$month_of_app = as.numeric(format(df$DateOfAppointment, "%m"))
df$status = ifelse(df$Status  == 'Show-Up',0,1)

library(ggplot2)

df %>%
  ggplot() +
  geom_histogram(aes(month_of_app, color = Status))
  
df %>%
  ggplot() +
  geom_histogram(aes(month_of_app_made, color = Status))

str(df)
train_chr_dt = df %>%
  select(-Status) %>%
  select(-ID) %>%
  select(-DateAppointmentWasMade) %>%
  select(-DateOfAppointment) %>%
  select(-Gender) %>%
  select(-DayOfTheWeek)

str(train_chr)


## leave out the date data
train_chr = train_chr_dt %>%
  select(-year_of_app, -year_of_app_made, -month_of_app, -month_of_app_made)

day_week = model.matrix(~DayOfTheWeek-1, df)

train = cbind(train_chr,day_week)
train_numeric = data.frame(apply(train,2,as.numeric))

# prepare train and test data
data_set_size = floor(nrow(train_numeric)*.6)

indexes = sample(1:nrow(train_numeric), size = data_set_size)

train.s = train_numeric[indexes,]
test.s = train_numeric[-indexes,]

train_matrix = data.matrix(train.s[,-12])
test_matrix = data.matrix(test.s[,-12])

dtrain = xgb.DMatrix(data = train_matrix, label = train_numeric$status[indexes])
dtest = xgb.DMatrix(data = test_matrix, label = train_numeric$status[-indexes])

## use cv to choose parameters

tls = list()
grid = seq(1,20,length = 20)

for(i in 1:20){

  model = xgb.cv(data=dtrain, nfold = 10, max.depth = 18,nround = 18, objective = 'binary:logistic', eval_metric = 'logloss', gamma =grid[i],min_child_weight = 37)
  tls[[i]] = model$evaluation_log
  }

grid1 = seq(21,30,length = 20)
tls1 =list()

for(i in 1:20){
  
  model = xgb.cv(data=dtrain, nfold = 10, max.depth = 18,nround = 18, objective = 'binary:logistic', eval_metric = 'logloss', gamma =grid1[i],min_child_weight = 37)
  tls1[[i]] = model$evaluation_log
}

grid2 = seq(31,45,length = 20)
tls2 =list()

for(i in 1:20){
  
  model = xgb.cv(data=dtrain, nfold = 10, max.depth = 20,nround = 10, objective = 'binary:logistic', eval_metric = 'logloss', gamma =grid2[i],min_child_weight = 37)
  tls2[[i]] = model$evaluation_log
}

grid3 = seq(25,33,length = 20)
tls3 = list()
for(i in 1:20){
  
  model = xgb.cv(data=dtrain, nfold = 10, max.depth = 20,nround = 10, objective = 'binary:logistic', eval_metric = 'logloss', gamma =grid3[i],min_child_weight = 37)
  tls3[[i]] = model$evaluation_log
}

tls3_min = matrix(NA,20,2)

for(i in 1:20){
  tls3_min[i,1] = i
  tls3_min[i,2] = tls3[[i]][,4] %>% min()
}

min(tls3_min[,2])

grid3[1]

grid4 = seq(23,26,length = 10)
tls4 = list()

for(i in 1:10){
  
  model = xgb.cv(data=dtrain, nfold = 10, max.depth = 20,nround = 15, objective = 'binary:logistic', eval_metric = 'logloss', gamma =grid4[i],min_child_weight = 37)
  tls4[[i]] = model$evaluation_log
}

tls4_min = matrix(NA,10,2)

for(i in 1:10){
  tls4_min[i,1] = i
  tls4_min[i,2] = tls4[[i]][,4] %>% min()
}

min(tls4_min[,2])

grid5 = seq(20,23,length = 10)
tls5 = list()

for(i in 1:10){
  
  model = xgb.cv(data=dtrain, nfold = 10, max.depth = 20,nround = 15, objective = 'binary:logistic', eval_metric = 'logloss', gamma =grid5[i],min_child_weight = 37)
  tls5[[i]] = model$evaluation_log
}

tls5_min = matrix(NA,10,2)

for(i in 1:10){
  tls5_min[i,1] = i
  tls5_min[i,2] = tls5[[i]][,4] %>% min()
}

min(tls5_min[,2])

grid5[5]

xgb.cv(data=dtrain, nfold = 10, max.depth = 20,nround = 17, objective = 'binary:logistic', eval_metric = 'logloss', gamma =21.3333, min_child_weight = 40)

## choose min_child_weight

grid6 = seq(35,50,length = 15)
tls6 = list()

for(i in 1:15){
  
  model = xgb.cv(data=dtrain, nfold = 10, max.depth = 20,nround = 15, objective = 'binary:logistic', eval_metric = 'logloss', gamma = 21.3333,min_child_weight = grid6[i])
  tls6[[i]] = model$evaluation_log
}

tls6_min = matrix(NA,10,2)

for(i in 1:10){
  tls6_min[i,1] = i
  tls6_min[i,2] = tls6[[i]][,4] %>% min()
}

min(tls6_min[,2])
grid6[6]

xgb.cv(data=dtrain, nfold = 10, max.depth = 20, nround = 17, objective = 'binary:logistic', eval_metric = 'logloss', gamma = 21, min_child_weight = 40)

model.xgb = xgboost(data=dtrain, nfold = 10, max.depth = 20, nround = 17, objective = 'binary:logistic', eval_metric = 'logloss', gamma = 13, min_child_weight = 37)

pred = predict(model.xgb, dtest)
prediction = as.numeric(pred > 0.5)

mean(prediction != test.s[,12])

## function that make prediction

pred_data <- read_csv("~/Dropbox/Predict_NoShow_PrivateTest_WithoutLabels.csv")

pred_df = pred_data
pred_df$Gender_n = ifelse(pred_df$Gender=='M',1,0)

pred_df = pred_df %>%
  select(-ID, -DateAppointmentWasMade, -DateOfAppointment, -Gender, -DayOfTheWeek)

pred_day_week = model.matrix(~DayOfTheWeek-1, pred_data)

pred_df1 = cbind(pred_df,pred_day_week)
pred_numeric = data.frame(apply(pred_df1,2,as.numeric))
str(pred_df1)
unique(pred_data$DayOfTheWeek)

pred_numeric$DayOfTheWeekSunday = 0
pred_matrix = data.matrix(pred_numeric)

dpred = xgb.DMatrix(data = pred_matrix)

pred.private = predict(model.xgb,dpred)

predictino.pr = data.frame(cbind(pred_data$ID,pred.private))


pred_pr = read_csv("~/Dropbox/Predict_NoShow_PrivateTest_WithoutLabels.csv")
pred_pub = read_csv('~/Dropbox/Predict_NoShow_PublicTest_WithoutLabels.csv')

predict.xgb = function(pred_data){

  pred_df = pred_data
  pred_df$Gender_n = ifelse(pred_df$Gender=='M',1,0)
  
  pred_df = pred_df %>%
    select(-ID, -DateAppointmentWasMade, -DateOfAppointment, -Gender, -DayOfTheWeek)
  
  pred_day_week = model.matrix(~DayOfTheWeek-1, pred_data)
  
  pred_df1 = cbind(pred_df,pred_day_week)
  
  pred_numeric = data.frame(apply(pred_df1,2,as.numeric))
  
  pred_numeric$DayOfTheWeekSunday = 0
  pred_matrix = data.matrix(pred_numeric)
  
  dpred = xgb.DMatrix(data = pred_matrix)
  
  pred = predict(model.xgb,dpred)
  
  prediction = data.frame(cbind(pred_data$ID,pred))
  colnames(prediction)[1] = 'ID'
  
  return(prediction)
}

write_csv(data.frame(predict.xgb(pred_pr)),'private.csv',col_names = FALSE)
write_csv(data.frame(predict.xgb(pred_pub)),'public.csv',col_names = FALSE)

