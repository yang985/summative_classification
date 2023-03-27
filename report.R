#author: "qlvv56"

hotels <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/hotels.csv")

library("skimr")
#skim(hotels)
library("tidyverse")
library("ggplot2")
library("DataExplorer")

DataExplorer::plot_bar(hotels, ncol = 1,nrow = 2)

ggplot(hotels,
       aes(x = adr, y = stays_in_weekend_nights+stays_in_week_nights)) +
  geom_point()
# one set of data with expensive booking value may affect the training result so we decided to filter it from the dataset 

hotels <- hotels |>
  filter(adr < 4000) |> 
  mutate(total_nights = stays_in_weekend_nights+stays_in_week_nights)

#eliminating variables
hotels <- hotels |>
  select(-reservation_status, -reservation_status_date) |> 
  mutate(kids = case_when(
    children + babies > 0 ~ "kids",
    TRUE ~ "none"
  ))

hotels <- hotels |> 
  select(-babies, -children)

hotels <- hotels |> 
  mutate(parking = case_when(
    required_car_parking_spaces > 0 ~ "parking",
    TRUE ~ "none"
  )) |> 
  select(-required_car_parking_spaces)



# logistic regression model
library("skimr")
#skim(hotels)
set.seed(6)
# Remove all categorical variables with 10 or more unique levels, except for arrival_date_month.

hotels_lr <- hotels |> 
  select(-country, -reserved_room_type, -assigned_room_type, -agent, -company,
         -stays_in_weekend_nights, -stays_in_week_nights)

#train/test/validate strategy
require(caTools)
ind <- sample.split(Y=hotels_lr$is_canceled,SplitRatio = 0.7)
train_lr = hotels_lr[ind,]
test_lr = hotels_lr[!ind,]

fit.lr <- glm(as.factor(is_canceled) ~ ., binomial, train_lr)
#summary(fit.lr)

pred.lr.res <- predict(fit.lr, test_lr, type = "response")

ggplot(data.frame(x = pred.lr.res), aes(x = x)) + geom_histogram()

predictions_lr <- ifelse(pred.lr.res > 0.5,1,0)

require(caret)
confusionMatrix(as.factor(predictions_lr), as.factor(test_lr$is_canceled))

library(classifierplots)
classifierplots::calibration_plot(test_lr$is_canceled,pred.lr.res)


# linear discriminant analysis empplymentation
require(MASS)
set.seed(6)
# dumming coding for variables
require('fastDummies')

varsForDumming <- filter(skim(train_lr),skim_type=="character")$skim_variable
hotels_lr_dumming <- dummy_cols(hotels_lr, select_columns = varsForDumming,
                                remove_selected_columns = TRUE)

train_lr_dumming = hotels_lr_dumming[ind,]
test_lr_dumming = hotels_lr_dumming[!ind,]

fit.lda <- MASS::lda(as.factor(is_canceled) ~ ., train_lr_dumming)

pred.lda.res <- predict(fit.lda, test_lr_dumming, type = "response")

ldahist(data = pred.lda.res$x[,1], g = test_lr_dumming$is_canceled)
#ggplot(data.frame(x = pred.lda.res), aes(x = x)) + geom_histogram()

#base::mean(I(pred.lda.res$class == na.omit(test_lr_dumming$is_canceled)))
require(caret)
confusionMatrix(as.factor(pred.lda.res$class),as.factor(test_lr_dumming$is_canceled))


# PCA
hotels_lr_dumming
scaled_hotel <- scale(hotels_lr_dumming[,2:57])
pca_result <- prcomp(scaled_hotel)
summ_pca <- summary(pca_result)
summ_pca$importance

cumulative_prop <- summ_pca$importance[3,1:56]
x <- 1:56
plot(x, cumulative_prop, type="b", main = "Screeplot of cumulative proportion")


# KNN algorithem


require(class)
set.seed(6)
my_knn_model <- class::knn(train_lr_dumming[,2:36],test_lr_dumming[,2:36],train_lr_dumming$is_canceled,k=3)

# In this case we used 4 default values 

require(caret)
confusionMatrix(as.factor(my_knn_model),as.factor(test_lr_dumming$is_canceled))
