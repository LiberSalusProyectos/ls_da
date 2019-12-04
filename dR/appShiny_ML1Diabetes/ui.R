#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(kernlab)

shinyUI(fluidPage(

    # Application title
    titlePanel("Formulario de datos"),

    sidebarLayout(
        sidebarPanel(
            numericInput("pregnancies", "Número de embarazos:", value = 2, min = 0, max = 17, step = 1),
            numericInput("glucose", "Concentración plasmática de glucosa a 2 horas en prueba oral de tolerancia a glucosa:", value = 2, min = 0, max = 199, step = 1),
            numericInput("bloodPressure", "Presión arterial diastólica:", value = 2, min = 0, max = 122, step = 1),
            numericInput("skinThickness", "Espesor de los pliegues de la piel del tríceps:", value = 2, min = 0, max = 99, step = 1),
            numericInput("insulin", "Insulina en suero 2hr:", value = 2, min = 0, max = 846, step = 1),
            numericInput("bmi", "Índice de masa corporal:", value = 2, min = 0, max = 67.1, step = 0.1),
            numericInput("diabetesPedigreeFunction", "Función genealógica de diabetes:", value = 2, min = 0, max = 2.42, step = 0.01),
            numericInput("age", "Edad:", value = 2, min = 6, max = 99, step = 1),
            submitButton("Enviar")
        ),
        

        mainPanel(
            textOutput("prediccion")
        )
    )
))
