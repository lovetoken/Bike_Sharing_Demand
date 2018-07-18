## ----env_ready, echo = F, warning = F, message = F-----------------------
pacman::p_load(knitr, tidyverse, tidyr, ggplot2, data.table, caret, magrittr, lubridate, MLmetrics)
opts_chunk$set(fig.path = "output/figure/", fig.align = "center", out.width = "90%", warning = F, message = F)

data_path = "data/"

## ----Rawdata_read--------------------------------------------------------
d <- paste0(data_path, "train.csv") %>% read_csv

## ----Preprocessing_&_Partition-------------------------------------------
pre_d <- d %>% 
  mutate(month = month(datetime),
         wday = wday(datetime, label = T),
         hour = hour(datetime)) %>% 
  mutate_at(vars(month, wday, hour, season, holiday, workingday, weather), as.factor) %>% 
  select(-datetime, -casual, -registered) %>% 
  select(month, wday, hour, everything())

index <- createDataPartition(pre_d$count, p = .8, list = F)
train <- pre_d %>% extract(index, )
test <- pre_d %>% extract(-index, )

## ----Modeling------------------------------------------------------------
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = TRUE)
mlMethods <- c("glm", "simpls", "rpart", "pcr", "kknn", "glmnet", "lasso", "ridge")
models <- mlMethods %>% 
  lapply(function(x) train(count ~ ., data = train, method = x, trControl = fitControl))

## ----evaluate_function_setting-------------------------------------------
evaluate <- function(model, testset, class, ylim = c(1, testset %>% pull(class) %>% max)){
  stopifnot(is.character(class))

  pd <- data.frame(real = pull(testset, class), 
                   pred = predict(model, newdata = testset) %>% unlist) %>% tbl_df %>% 
    arrange(real) %>% 
    mutate(index = 1:nrow(.)) %>% 
    gather(class, value, -index)
    
  p <- ggplot(pd, aes(x = index, y = value, color = class)) + 
    geom_line(size = .3, alpha = .7) + 
    ggtitle(model$modelInfo$label, 
            paste0("process time : ", model$times$everything, " / ", 
                   "RMSE : ", RMSE(predict(model, newdata = testset) %>% unlist, pull(testset, class))))
  p
}

## ----Evaluate------------------------------------------------------------
models %>% 
  lapply(evaluate, test, "count")

bestModel <- models %>% 
  lapply(function(x) RMSE(predict(x, newdata = test) %>% unlist, pull(test, count))) %>% 
  unlist %>% 
  order %>% extract(1) %>% 
  extract2(models, .)

## ----Predict_on_real_testset---------------------------------------------
realTest <- paste0(data_path, "test.csv") %>% read_csv 
predY <- realTest %>% 
  mutate(month = month(datetime),
         wday = wday(datetime, label = T),
         hour = hour(datetime)) %>% 
  mutate_at(vars(month, wday, hour, season, holiday, workingday, weather), as.factor) %>% 
  select(month, wday, hour, everything()) %>% 
  predict(bestModel, newdata = .)

## ----Output_submission---------------------------------------------------
submissionSet <- paste0(data_path, "sampleSubmission.csv") %>% 
  read_csv %>% 
  mutate(count = predY)

write_csv(submissionSet, "output/submissionSet.csv")

