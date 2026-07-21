open Bigarray

module type TENSOR = sig
  exception Shape_mismatch
  type t = {
    shape : int array;
    strides : int array;
    storage : (float, float32_elt, c_layout) Bigarray.Array1.t;
  }

  (* Helpers *)
  val len : t -> int

  (* Initialization *)
  val empty : int array -> t
  val zeros : int array -> t
  val of_data : int array -> float array -> t
  val shape : t -> int array
  val random: int array -> float * float -> int -> t

  (* Operators *)
  val (+): t -> t -> t
end

module Tensor : TENSOR = struct
  exception Shape_mismatch

  type t = {
    shape : int array;
    strides : int array;
    storage : (float, float32_elt, c_layout) Bigarray.Array1.t;
  }

  let shape t = t.shape

  let len t = Bigarray.Array1.dim t.storage

  let create input_shape =
    let len = Array.length input_shape in
    let current_product = ref 1 in
    let strides = Array.make len 1 in

    for i = len - 1 downto 0 do
      strides.(i) <- !current_product;
      current_product := !current_product * input_shape.(i)
    done;

    let total_size = !current_product in
    let storage = Array1.create float32 c_layout total_size in

    { shape = input_shape; strides; storage }

  let empty input_shape = create input_shape

  let zeros input_shape =
    let t = create input_shape in
    Array1.fill t.storage 0.0;
    t

  let of_data input_shape data =
    let expected_size = Array.fold_left ( * ) 1 input_shape in
    let actual_size = Array.length data in
    if expected_size <> actual_size then raise Shape_mismatch;
    let t = create input_shape in
    for i = 0 to actual_size - 1 do
      Array1.set t.storage i data.(i)
    done;
    t

  let random input_shape (x, y) seed =
      Random.init seed;

      let min_val = min x y in
      let max_val = max x y in
      let span = max_val -. min_val in

      let size = Array.fold_left ( * ) 1 input_shape in
      let t = create input_shape in

      for i = 0 to size - 1 do
        let rand_val = min_val +. Random.float span in
        Array1.set t.storage i rand_val
      done;
      t

  let (+) tensor1 tensor2 =
    t

end
