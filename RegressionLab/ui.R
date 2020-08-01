library(shiny)

shinyUI(
    fluidPage(

    titlePanel("Regression Lab"),

    sidebarLayout(
        sidebarPanel(
            sliderInput("tests",
                        "Tests:",
                        min = 1,
                        max = 200,
                        value = 20),
            numericInput("samples",
                         "Samples:",
                         value = 100),
            numericInput("features",
                         "Features:",
                         value = 90),
            numericInput("informative",
                         "Informative Features:",
                         value = 10),
            sliderInput("noise",
                         "Noise:",
                         min = 0,
                         max = 5,
                         value = 0.5,
                         step = 0.1),
            checkboxInput("advanced",
                          "Advanced:"),
            conditionalPanel(
                condition = "input.advanced == true",
                sliderInput("rank",
                             "Rank:",
                             min = -1,
                             max = 0,
                             value = -1,
                             step = 1)
            ),
            checkboxInput("large_font",
                          "Large Font"),
            actionButton("do", "Execute Experiment")
        ),
        mainPanel(
            plotOutput("distPlot")
        )
    )
))
