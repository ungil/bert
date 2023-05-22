library(bert)

bert <- load_from_file()
    
if (is.null(bert)){
    temp <- tempfile()
    download.file("https://huggingface.co/skeskinen/ggml/resolve/main/all-MiniLM-L6-v2/ggml-model-q4_0.bin", temp)
    bert <- load_from_file(temp)
}

fortunes <- fortunes::read.fortunes()$quote

embeddings <- sapply(fortunes, encode, ctx=bert)

findFortune <- function(query, k=3){
    query <- encode(bert, query)
    x <- (t(query) %*% embeddings)
    top <- order(x, decreasing=TRUE)[1:k]
    data.frame(score=x[top], fortune=colnames(x)[top])
}

findFortune("animal")

