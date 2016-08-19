rm(list= ls());
library(readr);
library(biglm);
library(glmnet);

# set working directory
setwd("/Users/Nathan/Documents/2010.UIUC.Course Work/STATS 542/FinalProject/");

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

cat("performing linear regression\n");

# perform linear regression on the data, full model
lm.model      = lm(target ~ ., data=train.data);

lm.predict = predict(lm.model, newdata = test.data, type = "response");
submission$target = lm.predict;

submission$target[submission$target < 0] = 0;
submission$target[submission$target > 1] = 1;

# printing results
cat("saving the submission file\n")
write_csv(submission, "linear_regression_full.csv")

# ****************** #
# perform linear regression on the data, noncategorical
cat("reducing model to only noncategorical\n");
lm.model.noncategorical = lm(target ~ ., data=train.data[!categorical_variables]);

lm.predict.noncategorical = predict(lm.model.noncategorical, newdata = test.data[!categorical_test], type = "response");
submission$target = lm.predict.noncategorical;

submission$target[submission$target < 0] = 0;
submission$target[submission$target > 1] = 1;

# printing results
cat("saving the submission file\n")
write_csv(submission, "linear_regression_noncategorical.csv");

# ****************** #
# perform linear regression on the data, lasso
predictors = names(train.data) != "target";
train.data.matrix = data.matrix(train.data[predictors & !categorical_variables]);
train.data.matrix[is.na(train.data.matrix)] = -1;

test.data.matrix = data.matrix(test.data[!categorical_test]);
test.data.matrix[is.na(test.data.matrix)] = -1;

cat("reducing model to only lasso");
glmnet.model.lasso = glmnet(y = train.data$target, x=train.data.matrix, family="binomial");

glmnet.predict.lasso = predict(glmnet.model.lasso, newx = test.data.matrix, type = "response");
submission$target = glmnet.predict.lasso;

submission$target[submission$target < 0] = 0;
submission$target[submission$target > 1] = 1;

# printing results
cat("saving the submission file\n")
write_csv(submission, "linear_regression_lasso.csv");

# ****************** #
# perform linear regression on the data, ridge
cat("reducing model to only ridge");
glmnet.model.ridge = glmnet(y = train.data$target, x=train.data.matrix, family="binomial", alpha = 0);

glmnet.predict.ridge = predict(glmnet.model.ridge, newx = test.data.matrix, type = "response");
submission$target = glmnet.predict.ridge;

submission$target[submission$target < 0] = 0;
submission$target[submission$target > 1] = 1;

# printing results
cat("saving the submission file\n")
write_csv(submission, "linear_regression_ridge.csv");
