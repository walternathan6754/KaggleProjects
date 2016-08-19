data1 = read.csv("Dropbox/Stats542_FinalReport/linear_regression_full.csv", header = TRUE)
data2 = read.csv("Dropbox/Stats542_FinalReport/xgboost.csv", header = TRUE)
data3 = read.csv("Dropbox/Stats542_FinalReport/linear_regression_ridge.csv", header = TRUE)

library(readr)

submission=data2

temporary=(data1$target+data2$target)/2

for (i in 1:length(temporary)){
if(temporary[i] > 0.2){
  submission$target[i]=max(data1$target,data2$target)
} else{
  submission$target[i]=min(data1$target,data2$target)
}
}

write_csv(submission, "special_xgboost_biglm_ridge.csv")
 