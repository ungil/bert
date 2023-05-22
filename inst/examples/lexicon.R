bert <- bert::load_from_file()
    
if (is.null(bert)){
    temp <- tempfile()
    download.file("https://huggingface.co/skeskinen/ggml/resolve/main/all-MiniLM-L12-v2/ggml-model-f16.bin", temp)
    bert <- bert::load_from_file(temp)
}

embeddings <- bert::encode(bert, lexicon::sw_fry_1000, 8)

close <- function(target, k=5){
    if (isa(target, "character")){
        stopifnot(target %in% colnames(embeddings))
        target <- embeddings[,target]        
        offset <- 1
    }else{
        offset <- 0
    }
    tmp <- t(target) %*% embeddings
    tmp[,order(tmp, decreasing=TRUE)][(1:k)+offset]
}

close(embeddings[,"foot"]-embeddings[,"leg"]+embeddings[,"arm"])

close(embeddings[,"father"]-embeddings[,"man"]+embeddings[,"woman"])

close(embeddings[,"mother"]-embeddings[,"woman"]+embeddings[,"man"])

close(embeddings[,"parent"]+embeddings[,"woman"])

close(embeddings[,"parent"]+embeddings[,"man"])

close(embeddings[,"ship"])

close(embeddings[,"ship"]-embeddings[,"sea"])

close(embeddings[,"plane"]-embeddings[,"air"]+embeddings[,"water"])

close(embeddings[,"bird"]-embeddings[,"air"]+embeddings[,"water"])

