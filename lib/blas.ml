open Bigarray

external cblas_gemm :
  int -> int -> int ->
  (float, float32_elt, c_layout) Array1.t ->
  (float, float32_elt, c_layout) Array1.t ->
  (float, float32_elt, c_layout) Array1.t -> unit
  = "caml_cblas_gemm_bytecode" "caml_cblas_gemm"

external saxpy :
  int -> float ->
  (float, float32_elt, c_layout) Array1.t ->
  (float, float32_elt, c_layout) Array1.t ->
  unit = "caml_saxpy"
