open Bigarray

exception Shape_mismatch

type t = {
  shape : int array;
  strides : int array;
  offset : int;
  storage : (float, float32_elt, c_layout) Array1.t;
}

val shape : t -> int array
val len : t -> int
val empty : int array -> t
val zeros : int array -> t
val of_data : int array -> float array -> t
val random : int array -> float * float -> ?seed:int -> unit -> t
val slice : t -> (int * int) array -> t
val get_row : t -> int -> t
val get_col : t -> int -> t
val ( + ) : t -> t -> t
val ( - ) : t -> t -> t
val ( * ) : t -> t -> t
val matmul : t -> t -> t
