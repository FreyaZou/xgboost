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
train_chr = df %>%
  select(-Status) %>%
  select(-ID) %>%
  select(-DateAppointmentWasMade) %>%
  select(-DateOfAppointment) %>%
  select(-Gender) %>%
  select(-DayOfTheWeek)

str(train_chr)

day_week = model.matrix(~DayOfTheWeek-1, head(Predict_NoShow_Train))
train = cbind(train_chr,day_week)
train_numeric = data.frame(apply(train,2,as.numeric))

# prepare train and test data
data_set_size = floor(nrow(train_numeric)*.6)

indexes = sample(1:nrow(train_numeric), size = data_set_size)

train.s = train_numeric[indexes,]
test.s = train_numeric[-indexes,]

train_matrix = data.matrix(train.s[,-16])
test_matrix = data.matrix(test.s[,-16])

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
  
  model = xgb.cv(data=dtrain, nfold = 10, max.depth = 18,nround = 10, objective = 'binary:logistic', eval_metric = 'logloss', gamma =grid2[i],min_child_weight = 37)
  tls2[[i]] = model$evaluation_log
}