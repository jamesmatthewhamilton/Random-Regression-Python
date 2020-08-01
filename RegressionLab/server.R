library(shiny)
library(ggplot2)
library(reticulate)

use_python("/usr/bin/python3", required = FALSE)
source_python("RegressionTests.py")

shinyServer(function(input, output, session) {

    res <- eventReactive(input$do, {
        results <- supermassive_regression_test(
            tests=as.integer(input$tests),
            sample_start=as.integer(input$samples),
            feature_start=as.integer(input$features),
            informative_start=as.integer(input$informative),
            verbose=FALSE
        )
        results
    })

    output$distPlot <- renderPlot({
        results = res()

        p <- ggplot()
        p <- p + ggtitle("Regression Lab Plot")
        df <- data.frame(y = results[[1]][1, ],
                         x = unlist(results[[4]][2, ], use.names=FALSE))
        p <- p + geom_line(data=df, aes(x, y, color=results[[5]][1]))

        for(ii in 2:length(results[[5]])) {
            print(ii)
            df <- data.frame(y = results[[1]][ii, ],
                             x = unlist(results[[4]][2, ], use.names=FALSE))
            p <- p + geom_line(data=cbind(df, col=results[[5]][ii]), aes(x, y, color=col))
            print(results[[5]][ii])
            p <- p + scale_y_log10()
        }
        if(input$large_font) {
            p <- p + theme_classic(base_size = 20)
        }
        p <- p + labs(x="Features", y="Root Mean Error") 
        p <- p + labs(color='Legend') 
        print(p)
    })
    
    observe({
        temp_rank <- input$rank
        temp_features <- input$features
        if (temp_rank > temp_features) { temp_rank = temp_features}
        updateSliderInput(session,
                          "rank",
                          min = 0,
                          max =  temp_features,
                          value = temp_rank)
    })

})
