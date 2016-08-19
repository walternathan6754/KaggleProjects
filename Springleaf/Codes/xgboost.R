rm(list= ls());
library(readr);
library(xgboost);
library(glmnet);

# set working directory
setwd("/Users/Geordie/Documents/STAT_542/project/");

# read in the data
train.data = read_csv("train.csv");
test.data  = read_csv("test.csv");

# store test IDS
submission = data.frame(ID=test.data$ID)

# extract the response variable
train.y = train.data$target;

# remove the ID variable because it has no meaning
cat("removing ID\n");
train.data = subset(train.data, select=-c(ID,target)); 
test.data  = subset(test.data, select=-c(ID)); 

unique_values    = sapply(train.data, function(x) length(unique(x)));

# remove variables with only 1 value
cat("removing non unique variables\n");
train.data = subset(train.data, select=unique_values != 1);
test.data  = subset(test.data, select=unique_values != 1);

na_percent   = sapply(train.data, function(x) length(which(is.na(x)))/length(x));

# remove variables with more than 15% na values
cat("removing na variables\n");
train.data = subset(train.data, select=na_percent < .15);
test.data  = subset(test.data, select=na_percent < .15);

# remove variables 0212 0227 0228 as them seem to be ids
# and they appear to be duplicates of one another just with nas
cat("removing id variables\n");

unique_values    = sapply(train.data, function(x) length(unique(x)));

train.data = subset(train.data, select=unique_values < 100000);
test.data  = subset(test.data, select=unique_values < 100000);

cat("Make all text variables into numerical levels\n");
predictors = names(train.data)[2:ncol(train.data)-1];

# note which columns are categorical
train.data$target = train.y;
categorical_variables = sapply(train.data, is.character);
categorical_test = categorical_variables[names(categorical_variables) != "target"];

for (predictor in predictors) {
  if (class(train.data[[predictor]])=="character") {
    levels = unique(c(train.data[[predictor]], test.data[[predictor]]));
    train.data[[predictor]] = as.integer(factor(train.data[[predictor]], levels=levels));
    test.data[[predictor]]  = as.integer(factor(test.data[[predictor]],  levels=levels));
  }
}

# make NA values -1, this applies only to integer predictors, since
# categorical values should have already assessed NA as a level in
# previous section
cat("Make remaining NA values -1\n");
train.data[is.na(train.data)] = -1
test.data[is.na(test.data)]   = -1

cat(sprintf("Number of predictors reduced to %d\n", ncol(train.data)));

########################################################################
### Previous section should only need to be run once to extract data ###
########################################################################

names <- names(train.data)[2:ncol(train.data)-1]

cat("sampling train to get around 8GB memory limitations\n")
train.data <- train.data[sample(nrow(train.data), 40000),]
gc()

cat("training a XGBoost classifier\n")
classifier <- xgboost(data        = data.matrix(train.data[,names]),
                      label       = train.data$target,
                      missing     = NaN,
                      nrounds     = 20,
                      objective   = "binary:logistic",
                      eval_metric = "auc")

cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=test.data$ID)
submission$target <- NA 
for (rows in split(1:nrow(test.data), ceiling((1:nrow(test.data))/10000))) {
  submission[rows, "target"] <- predict(classifier, data.matrix(test[rows,names]))
}

# printing results
cat("saving the submission file\n")
write_csv(submission, "xgboost_submission.csv");
