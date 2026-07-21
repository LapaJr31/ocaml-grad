#include <caml/mlvalues.h>
#include <caml/bigarray.h>
#include <cblas.h>

CAMLprim value caml_cblas_gemm(
    value vM, value vN, value vK,
    value vA, value vB, value vC)
{
    int M = Int_val(vM);
    int N = Int_val(vN);
    int K = Int_val(vK);

    float *A = (float *)Caml_ba_data_val(vA);
    float *B = (float *)Caml_ba_data_val(vB);
    float *C = (float *)Caml_ba_data_val(vC);

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0f,        /* alpha */
        A, K,        /* A (M×K), lda = K */
        B, N,        /* B (K×N), ldb = N */
        0.0f,        /* beta  */
        C, N         /* C (M×N), ldc = N */
    );
    return Val_unit;
}

CAMLprim value caml_cblas_gemm_bytecode(value *argv, int argn) {
    return caml_cblas_gemm(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value caml_saxpy(
    value vN, value vAlpha, value vX, value vY)
{
    int N = Int_val(vN);
    float alpha = (float)Double_val(vAlpha);
    float *X = (float *)Caml_ba_data_val(vX);
    float *Y = (float *)Caml_ba_data_val(vY);

    cblas_saxpy(N, alpha, X, 1, Y, 1);
    return Val_unit;
}
