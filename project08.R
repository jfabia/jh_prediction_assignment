library(data.table)
library(caret)
library(AppliedPredictiveModeling)
library(rattle)
library(randomForest)
library(rpart.plot)
library(gbm)
library(survival)
library(splines)
library(parallel)
library(plyr)

rm(list=ls())

datasource.train <- "./pml-training.csv"
datasource.test <- "./pml-testing.csv"
data.train.URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
data.test.URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Check if both files exist.
if( !file.exists(datasource.train)){
        download.file(data.train.URL, destfile = "./pml-training.csv")
        }
if( !file.exists(datasource.test)){
        download.file(data.test.URL, destfile = "./pml-testing.csv")
}        


# Load the data and manual from their respective sources
project.data.train <- read.csv(datasource.train)
project.data.test <- read.csv(datasource.test)

# There are many columns with blank entries or NA entries, so we drop them
# from our analysis.

clean.train <- 
        subset(project.data.train, 
               select=c("roll_belt","pitch_belt","yaw_belt",
                        "total_accel_belt",
                        "gyros_belt_x","gyros_belt_y","gyros_belt_z",
                        "accel_belt_x","accel_belt_y","accel_belt_z",
                        "magnet_belt_x","magnet_belt_y","magnet_belt_z",
                        "roll_arm","pitch_arm","yaw_arm",
                        "total_accel_arm",
                        "gyros_arm_x","gyros_arm_y","gyros_arm_z",
                        "accel_arm_x","accel_arm_y","accel_arm_z",
                        "magnet_arm_x","magnet_arm_y","magnet_arm_z",
                        "roll_dumbbell","pitch_dumbbell","yaw_dumbbell",
                        "total_accel_dumbbell",
                        "gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z",
                        "accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z",
                        "magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z",
                        "roll_forearm","pitch_forearm","yaw_forearm",
                        "total_accel_forearm",
                        "gyros_forearm_x","gyros_forearm_y","gyros_forearm_z",
                        "accel_forearm_x","accel_forearm_y","accel_forearm_z",
                        "magnet_forearm_x","magnet_forearm_y","magnet_forearm_z",
                        "classe"))



clean.test <- 
        subset(project.data.test, 
               select=c("roll_belt","pitch_belt","yaw_belt",
                        "total_accel_belt",
                        "gyros_belt_x","gyros_belt_y","gyros_belt_z",
                        "accel_belt_x","accel_belt_y","accel_belt_z",
                        "magnet_belt_x","magnet_belt_y","magnet_belt_z",
                        "roll_arm","pitch_arm","yaw_arm",
                        "total_accel_arm",
                        "gyros_arm_x","gyros_arm_y","gyros_arm_z",
                        "accel_arm_x","accel_arm_y","accel_arm_z",
                        "magnet_arm_x","magnet_arm_y","magnet_arm_z",
                        "roll_dumbbell","pitch_dumbbell","yaw_dumbbell",
                        "total_accel_dumbbell",
                        "gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z",
                        "accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z",
                        "magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z",
                        "roll_forearm","pitch_forearm","yaw_forearm",
                        "total_accel_forearm",
                        "gyros_forearm_x","gyros_forearm_y","gyros_forearm_z",
                        "accel_forearm_x","accel_forearm_y","accel_forearm_z",
                        "magnet_forearm_x","magnet_forearm_y","magnet_forearm_z"))

# belt, arm, dumbbell and forearm pertain to where the trackers are.
# Hence we have Belt/Arm/Dumbbell/Forearm as the first part of the variable.

# Next, we have roll, pitch and yaw, which refer to the rotation axes.
# This will be the second name of our variable

# gyros, accel and magnet refer to the inertial measurement unit tracking
# Specific Force, Angular Rate and Magnetic Field surrounding the body,
# using a combination of Gyroscopes, Accelerometers and Magnetometers. 
# For purpose of this study we will rename Gyros, Accel and Magnet to
# AngularRate, SpecificForce, and MagneticField.
# We will also name the axes of X, Y and Z as X,Y and Z.


names(clean.train) <- c("Belt.Roll","Belt.Pitch","Belt.Yaw",
                        "Belt.TotalSpecificForce","Belt.AngularRate.Xaxis",
                        "Belt.AngularRate.Yaxis",
                        "Belt.AngularRate.Zaxis","Belt.SpecificForce.Xaxis",
                        "Belt.SpecificForce.Yaxis",
                        "Belt.SpecificForce.Zaxis",
                        "Belt.MagneticField.Xaxis",
                        "Belt.MagneticField.Yaxis",
                        "Belt.MagneticField.Zaxis",
                        "Arm.Roll","Arm.Pitch","Arm.Yaw",
                        "Arm.TotalSpecificForce",
                        "Arm.AngularRate.Xaxis",
                        "Arm.AngularRate.Yaxis",
                        "Arm.AngularRate.Zaxis",
                        "Arm.SpecificForce.Xaxis",
                        "Arm.SpecificForce.Yaxis",
                        "Arm.SpecificForce.Zaxis",
                        "Arm.MagneticField.Xaxis",
                        "Arm.MagneticField.Yaxis",
                        "Arm.MagneticField.Zaxis",
                        "Dumbbell.Roll","Dumbbell.Pitch","Dumbbell.Yaw",
                        "Dumbbell.TotalSpecificForce",
                        "Dumbbell.AngularRate.Xaxis",
                        "Dumbbell.AngularRate.Yaxis",
                        "Dumbbell.AngularRate.Zaxis",
                        "Dumbbell.SpecificForce.Xaxis",
                        "Dumbbell.SpecificForce.Yaxis",
                        "Dumbbell.SpecificForce.Zaxis",
                        "Dumbbell.MagneticField.Xaxis",
                        "Dumbbell.MagneticField.Yaxis",
                        "Dumbbell.MagneticField.Zaxis",
                        "Forearm.Roll","Forearm.Pitch","Forearm.Yaw",
                        "Forearm.TotalSpecificForce",
                        "Forearm.AngularRate.Xaxis",
                        "Forearm.AngularRate.Yaxis",
                        "Forearm.AngularRate.Zaxis",
                        "Forearm.SpecificForce.Xaxis",
                        "Forearm.SpecificForce.Yaxis",
                        "Forearm.SpecificForce.Zaxis",
                        "Forearm.MagneticField.Xaxis",
                        "Forearm.MagneticField.Yaxis",
                        "Forearm.MagneticField.Zaxis",
                        "Class")


names(clean.test) <- c("Belt.Roll","Belt.Pitch","Belt.Yaw",
                       "Belt.TotalSpecificForce","Belt.AngularRate.Xaxis",
                       "Belt.AngularRate.Yaxis",
                       "Belt.AngularRate.Zaxis","Belt.SpecificForce.Xaxis",
                       "Belt.SpecificForce.Yaxis",
                       "Belt.SpecificForce.Zaxis",
                       "Belt.MagneticField.Xaxis",
                       "Belt.MagneticField.Yaxis",
                       "Belt.MagneticField.Zaxis",
                       "Arm.Roll","Arm.Pitch","Arm.Yaw",
                       "Arm.TotalSpecificForce",
                       "Arm.AngularRate.Xaxis",
                       "Arm.AngularRate.Yaxis",
                       "Arm.AngularRate.Zaxis",
                       "Arm.SpecificForce.Xaxis",
                       "Arm.SpecificForce.Yaxis",
                       "Arm.SpecificForce.Zaxis",
                       "Arm.MagneticField.Xaxis",
                       "Arm.MagneticField.Yaxis",
                       "Arm.MagneticField.Zaxis",
                       "Dumbbell.Roll","Dumbbell.Pitch","Dumbbell.Yaw",
                       "Dumbbell.TotalSpecificForce",
                       "Dumbbell.AngularRate.Xaxis",
                       "Dumbbell.AngularRate.Yaxis",
                       "Dumbbell.AngularRate.Zaxis",
                       "Dumbbell.SpecificForce.Xaxis",
                       "Dumbbell.SpecificForce.Yaxis",
                       "Dumbbell.SpecificForce.Zaxis",
                       "Dumbbell.MagneticField.Xaxis",
                       "Dumbbell.MagneticField.Yaxis",
                       "Dumbbell.MagneticField.Zaxis",
                       "Forearm.Roll","Forearm.Pitch","Forearm.Yaw",
                       "Forearm.TotalSpecificForce",
                       "Forearm.AngularRate.Xaxis",
                       "Forearm.AngularRate.Yaxis",
                       "Forearm.AngularRate.Zaxis",
                       "Forearm.SpecificForce.Xaxis",
                       "Forearm.SpecificForce.Yaxis",
                       "Forearm.SpecificForce.Zaxis",
                       "Forearm.MagneticField.Xaxis",
                       "Forearm.MagneticField.Yaxis",
                       "Forearm.MagneticField.Zaxis")


# Create a counter that will divide the training dataset into a trained dataset
# and test dataset.
set.seed(1234567890)
indexTrain <- createDataPartition(clean.train$Class, p = .70)[[1]]
train.training <- clean.train[ indexTrain, ]
train.testing <- clean.train[ -indexTrain, ]
train.seed <- 987654321

# Reset the seed
set.seed(train.seed)
# Train data using Classification tree with normalized variables and cross-validation with 3-folds
rpart.model <- train(Class ~ . , data=train.training, method = "rpart", 
                        preProcess=c("center","scale"),
                        trControl = trainControl(method="cv", number=3))
# Plot used for tree
fancyRpartPlot(rpart.model$finalModel, 
               main = "",
               sub = "")
# Predict using Classification tree
pred.rpart <- predict(rpart.model, train.testing)
# Print the prediction model and confusion matrix
print(rpart.model)
confusionMatrix(pred.rpart, train.testing$Class)

# Reset the 
set.seed(train.seed)

# Train data using Random Forest with normalized variables and cross-validation with 3-folds
rf.model <- randomForest(Class ~ . , data=train.training, importance=TRUE,
                            trControl = trainControl(method="cv", number=3),
                            preProcess=c("center","scale"))
# Print the model
print(rf.model, digits=3)
# Predict using the random forest and test data
pred.rf <- predict(rf.model, train.testing)
# Print the confusion matrix
confusionMatrix(pred.rf, train.testing$Class)

# Reset the seed
set.seed(train.seed)
# Train data using Boosted tree with normalized variables and cross-validation with 3 folds
gbm.model <- train(Class ~ ., data=train.training, method = "gbm",
                      verbose = FALSE,
                      trControl = trainControl(method="cv", number=3),
                      preProcess = c("center","scale"))
# Pring the model
print(gbm.model, digits = 3)
# Predict using the boosted tree model and test data
pred.gbm <- predict(gbm.model, train.testing)
# Print the confusion matrix
confusionMatrix(pred.gbm, train.testing$Class)

