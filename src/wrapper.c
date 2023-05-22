#include <R.h>
#include <Rinternals.h>

#include "bert.h"

SEXP load_from_file(SEXP model);

SEXP encode(SEXP ctx, SEXP text, SEXP nthreads);

static void ctx_finalizer(SEXP ptr);

SEXP load_from_file(SEXP model){
  const char *filename = CHAR(STRING_ELT(model, 0));
  struct bert_ctx *ctx;
  ctx = bert_load_from_file(filename);
  if (ctx == NULL) {
    fprintf(stderr, "failed to load model\n");
    return R_NilValue;
  }
  SEXP out, ptr, names;
  out = PROTECT(allocVector(INTSXP, 2));
  ptr = R_MakeExternalPtr(ctx, install("bert_context"), R_NilValue);
  PROTECT(ptr);
  R_RegisterCFinalizerEx(ptr, ctx_finalizer, TRUE);
  INTEGER(out)[0] = bert_n_max_tokens(ctx);
  INTEGER(out)[1] = bert_n_embd(ctx);
  PROTECT(names = allocVector(STRSXP, 2));
  SET_STRING_ELT(names, 0, mkChar("n_max_tokens"));
  SET_STRING_ELT(names, 1, mkChar("n_embd"));
  setAttrib(out, R_NamesSymbol, names);
  setAttrib(out, install("bert_context_ptr"), ptr);
  UNPROTECT(3);
  return out;
}
 
static void ctx_finalizer(SEXP ptr)
{
  if(!R_ExternalPtrAddr(ptr)) return;
  bert_free(R_ExternalPtrAddr(ptr));
  R_ClearExternalPtr(ptr);
}

SEXP encode(SEXP ctx, SEXP text, SEXP threads){
  int nthreads=INTEGER(threads)[0];
  struct bert_ctx *ptr = R_ExternalPtrAddr(getAttrib(ctx, install("bert_context_ptr")));
  float *output;
  int n=bert_n_embd(ptr);
  output = malloc(n * sizeof(float));
  bert_encode(ptr, nthreads, CHAR(STRING_ELT(text, 0)), output);
  SEXP out = PROTECT(allocVector(REALSXP, n));
  for (int i=0; i<n; i++) REAL(out)[i] = output[i];
  UNPROTECT(1);
  return out;
}

// register
 
#include <R_ext/Rdynload.h>

#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

static const R_CallMethodDef R_CallDef[] = {
  CALLDEF(load_from_file, 1),
  CALLDEF(encode, 2),
  {NULL, NULL, 0}
};

void
#ifdef HAVE_VISIBILITY_ATTRIBUTE
__attribute__ ((visibility ("default")))
#endif
R_init_wrapper(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, R_CallDef, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}

