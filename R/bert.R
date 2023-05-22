#' @useDynLib bert, .registration=TRUE, .fixes="C_"
NULL

#' @export
load_from_file <- function(model="~/bert.cpp/models/all-MiniLM-L12-v2/ggml-model-f16.bin"){
    model <- path.expand(model)
    if (file.exists(model))
        .Call("load_from_file", model)
}

#' @export
encode <- function(ctx, text, nthreads=4){
    if (!is.null(ctx) && isa(attributes(ctx)$bert_context_ptr, "externalptr"))
        sapply(text, function(x) .Call("encode", ctx, x, as.integer(nthreads)))
}
