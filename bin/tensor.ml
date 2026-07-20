open Bigarray

module type TENSOR = sig
  exception Shape_mismatch
  type t = {
    shape : int array;
    strides : int array;
    storage : (float, float32_elt, c_layout) Bigarray.Array1.t;
  }
  val empty : int array -> t
  val zeros : int array -> t
  val of_data : int array -> float array -> t
end

module Tensor : TENSOR = struct
  type t = {
    shape : int array;
    strides : int array;
    storage : (float, float32_elt, c_layout) Bigarray.Array1.t;
  }

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

    { shape = input_shape; strides = strides; storage = storage }


  let empty input_shape =
    create input_shape

  let zeros input_shape =
    let t = create input_shape in
    Array1.fill t.storage 0.0;
    t

  exception Shape_mismatch
  let of_data input_shape data =
    let expected_size = Array.fold_left ( * ) 1 input_shape in
    let actual_size = Array.length data in
    if expected_size <> actual_size then
      raise Shape_mismatch;
    let t = create input_shape in
    Array1.blit (Bigarray.Array1.of_array Bigarray.float32 c_layout data) t.storage;

    t

end
