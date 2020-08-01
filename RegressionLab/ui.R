library(shiny)

shinyUI(
    fluidPage(

    titlePanel("Regression Lab"),

    sidebarLayout(
        sidebarPanel(
            tabsetPanel(type = "tabs",
            tabPanel("Setup",
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
            sliderInput("rank",
                        "Rank:",
                        min = 0,
                        max = 90,
                        value = 90,
                        step = 1),
            actionButton("do", "Execute Experiment")
            ),

            tabPanel("Change Function",
            h3("Which parameter should change?"),
            textInput("f_samples", h5("Samples:"),
                      value = "s"),
            textInput("f_features", h5("Features:"),
                     value = "f + x"),
            textInput("f_informative", h5("Informative:"),
                      value = "i"),
            textInput("f_noise", h5("Noise:"),
                      value = "n"),
            textInput("f_rank", h5("Rank:"),
                      value = "r"),
            ),

            tabPanel("Appearance",
            selectInput("yaxis", h3("Select box"), 
                        choices = list("Samples" = 1,
                                       "Features" = 2,
                                       "Informative" = 3,
                                       "Noise" = 4,
                                       "Rank" = 5),
                        selected = 2),        
            checkboxInput("large_font",
                          "Large Font"),
            ))
        ),
        mainPanel(
            plotOutput("distPlot")
        )
    )
))
