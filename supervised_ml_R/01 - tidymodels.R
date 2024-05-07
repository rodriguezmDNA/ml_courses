# https://supervised-ml-course.netlify.app


#install.packages("ranßßdomForest")
#install.packages('tidymodels')



## https://supervised-ml-course.netlify.app/chapter1
setwd("/Users/jrm/machine_learning/supervised_ml_R")

library(tidyverse)

cars2018 <- read_csv("data/cars2018.csv")
glimpse(cars2018)

# Plot the histogram
ggplot(cars2018, aes(x = mpg)) +
  geom_histogram(bins = 25) +
  labs(x = "Fuel efficiency (mpg)",
       y = "Number of cars")

# Build a simple linear model


##########
# There is a discrepancy in mpg values between the rds and the CSV. Use the rds
# version for reproducibility

## Deselect the 2 columns to create cars_vars
# car_vars_local <- cars2018 %>%
#   select(-model, -model_index)
#####

car_vars <- readRDS("data/c1_car_vars.rds")
##########


########## 
# Fit a linear model & print summary

fit_all <- lm(mpg ~ ., data = car_vars)
summary(fit_all)


### Training data and testing data with rsample

# Training data and testing data
# Build your model with your training data
# Choose your model with your validation data, or resampled datasets
# Evaluate your model with your testing data


library(tidymodels)

car_split <- car_vars %>%
  initial_split(prop = 0.8,
                strata = aspiration)

car_train <- training(car_split)
car_test <- testing(car_split)


### Training a model
#######################################

## a linear regression model specification
lm_mod <- linear_reg() %>%
  set_engine("lm")

lm_fit <- lm_mod %>%
  fit(log(mpg) ~ ., 
      data = car_train)

## a random forest model specification
rf_mod <- rand_forest() %>%
  set_mode("regression") %>%
  set_engine("randomForest")

fit_rf <- rf_mod %>%
  fit(log(mpg) ~ ., 
      data = car_train)        

# Three concepts in specifying a model
# Model type
# Model mode
# Model engine

## Evaluating a model
# the yardstick package 📏

#######

# Training models based on all of your data at once is typically not a good choice.
# Instead, you can create subsets of your data that you use for different purposes, such as training your model and then testing your model.

# Creating training/testing splits reduces overfitting.
# When you evaluate your model on data that it was not trained on, you get a better estimate of how it will perform on new data.


glimpse(car_train)
glimpse(car_test)

##################
# Build a linear regression model specification
lm_mod <- linear_reg() %>%
  set_engine("lm")

# Train a linear regression model
fit_lm <- lm_mod %>%
  fit(log(mpg) ~ ., 
      data = car_train)

# Print the model object
fit_lm


##################
# Build a random forest model specification
rf_mod <- rand_forest() %>%
  set_engine("randomForest") %>%
  set_mode("regression")

# Train a random forest model
fit_rf <- rf_mod %>%
  fit(log(mpg) ~ ., 
      data = car_train)

# Print the model object
fit_rf


####### Evaluate model performance

# The fit_lm and fit_rf models you just trained are in your environment.
# It’s time to see how they did! 

# For regression models, we will focus on evaluating using the root mean squared error metric.
# This quantity is measured in the same units as the original data (log of miles per gallon, in our case).
# Lower values indicate a better fit to the data.
# It’s not too hard to calculate root mean squared error manually, but the yardstick package offers convenient functions for this and many other model performance metrics.


# # Function to read and print .rds file from a URL
# read_rds_from_github <- function(url) {
#   # Download the file temporarily
#   temp <- tempfile()
#   download.file(url, temp, mode = "wb")  # Ensure binary mode for Windows compatibility
#   
#   # Read the RDS file
#   data <- readRDS(temp)
#   
#   # Clean up the temporary file
#   unlink(temp)
# }
# 

car_train <- readRDS("data/c1_train.rds")
car_test <- readRDS("data/c1_test.rds")

fit_lm <- readRDS("data/c1_fit_lm.rds")
fit_rf <- readRDS("data/c1_fit_rf.rds")


# Create the new columns
results <- car_train %>%
  mutate(mpg = log(mpg)) %>%
  bind_cols(predict(fit_lm, car_train) %>%
              rename(.pred_lm = .pred)) %>%
  bind_cols(predict(fit_rf, car_train) %>%
              rename(.pred_rf = .pred))

# Evaluate the performance
metrics(results, truth = mpg, estimate = .pred_lm)
metrics(results, truth = mpg, estimate = .pred_rf)

################################################################################
## 10 Evaluating models with resampling

lm_mod <- linear_reg() %>%
  set_engine("lm")

rf_mod <- rand_forest() %>%
  set_engine("randomForest") %>%
  set_mode("regression")

## Create bootstrap resamples
car_boot <- bootstraps(car_train)

# Evaluate the models with bootstrap resampling
lm_res <- lm_mod %>%
  fit_resamples(
    log(mpg) ~ .,
    resamples = car_boot,
    control = control_resamples(save_pred = TRUE)
  )

rf_res <- rf_mod %>%
  fit_resamples(
    log(mpg) ~ .,
    resamples = car_boot,
    control = control_resamples(save_pred = TRUE)
  )

glimpse(rf_res)

###

lm_res <- readRDS("data/c1_lm_res.rds")
rf_res <- readRDS("data/c1_rf_res.rds")

results <-  bind_rows(lm_res %>%
                        collect_predictions() %>%
                        mutate(model = "lm"),
                      rf_res %>%
                        collect_predictions() %>%
                        mutate(model = "rf"))

results

results %>%
  ggplot(aes(`log(mpg)`, .pred)) +
  geom_abline(lty = 2, color = "gray50") +
  geom_point(aes(color = id), size = 1.5, alpha = 0.3, show.legend = FALSE) +
  geom_smooth(method = "lm") +
  facet_wrap(~ model)

