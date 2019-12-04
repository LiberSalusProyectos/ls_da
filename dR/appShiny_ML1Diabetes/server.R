library(shiny)
library(kernlab)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {

    output$pregnancies <- renderText({
        input$pregnancies
    })
    output$prediccion <- renderText({
        modelo <- readRDS("ML3Diabetes_SVM.RDS")
        Pregnancies <- input$pregnancies
        Glucose <- input$glucose
        BloodPressure <- input$bloodPressure
        SkinThickness <- input$skinThickness
        Insulin <- input$insulin
        BMI <- input$bmi
        DiabetesPedigreeFunction <- input$diabetesPedigreeFunction
        Age <- input$age
        obs <- data.frame(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        pred <- predict(modelo, obs, type = "probabilities")
        diabetesProbability <- pred[1,2]
        paste("Probabilidad de padecer diabetes:", 100*round(diabetesProbability, digits = 4),"%")
    })
    

})
