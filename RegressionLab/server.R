library(shiny)

shinyServer(function(input, output, session) {

    output$distPlot <- renderPlot({

        x    <- faithful[, 2]
        bins <- seq(min(x), max(x), length.out = input$tests + 1)

        hist(x, breaks = bins)

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
