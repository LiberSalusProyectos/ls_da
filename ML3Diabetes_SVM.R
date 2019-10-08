library(kernlab)
diabetes <- read.csv('C:/Users/Liber Proyectos/Downloads/diabetes.csv')
diabetes$Outcome <- as.factor(diabetes$Outcome)
train <- sample(1:nrow(diabetes), size = nrow(diabetes)*0.75)
svm.spline <- ksvm(Outcome ~ ., data = diabetes[train,], type = "C-svc", 
                   kernel = "splinedot", prob.model=TRUE, C = 10, scale = TRUE)
svm.spline
svm.bessel <- ksvm(Outcome ~ ., data = diabetes[train,], type = "C-svc", 
                   kernel = "besseldot", prob.model=TRUE, C = 10, scale = TRUE)
pred <- predict(svm.spline, diabetes[-train,])
paste("Error de test:", 100*mean(diabetes$Outcome[-train] != pred),"%")
table(prediccion = pred, valor_real = diabetes$Outcome[-train])

# setwd('G:/Mi unidad/Desarrollo/desarrolloR')
# saveRDS(svm.spline, file = "ML3Diabetes_SVM.RDS")
