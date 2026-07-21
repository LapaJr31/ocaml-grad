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

  (* Operators — Note spaces inside ( * ) *)
  val ( + ) : t -> t -> t
  val ( - ) : t -> t -> t
  val ( * ) : t -> t -> t
end

module Tensor : TENSOR = struct
  exception Shape_mismatch

  type t = {
    shape : int array;
    strides : int array;
    storage : (float, float32_elt, c_layout) Bigarray.Array1.t;
  }

  (* PUBLIC helper functions *)
  let shape t = t.shape

  let len t = Bigarray.Array1.dim t.storage

  (* PRIVATE helper functions *)
  let compute_strides shape =
    let len = Array.length shape in
    let current_product = ref 1 in
    let strides = Array.make len 1 in
    for i = len - 1 downto 0 do
      strides.(i) <- !current_product;
      current_product := !current_product * shape.(i)
    done;
    strides

  let create input_shape =
    let strides = compute_strides input_shape in
    let total_size = Array.fold_left ( * ) 1 input_shape in
    let storage = Array1.create float32 c_layout total_size in
    { shape = input_shape; strides; storage }

  let pad_shape_left target_ndim shape =
    let current_ndim = Array.length shape in
    let diff = target_ndim - current_ndim in
    if diff <= 0 then shape
    else
      Array.append (Array.make diff 1) shape

  let broadcast_shape s1 s2 =
    let n = Array.length s1 in
    let out = Array.make n 0 in
    for i = 0 to n - 1 do
      let d1 = s1.(i) in
      let d2 = s2.(i) in
      if d1 = d2 then
        out.(i) <- d1
      else if d1 = 1 then
        out.(i) <- d2
      else if d2 = 1 then
        out.(i) <- d1
      else
        raise Shape_mismatch
    done;
    out

  let flat_idx padded_shape real_strides multi_idx =
    let acc = ref 0 in
    for d = 0 to Array.length padded_shape - 1 do
      if padded_shape.(d) <> 1 then
        acc := !acc + (multi_idx.(d) * real_strides.(d))
    done;
    !acc

  let elementwise op tensor1 tensor2 =
    let shape1 = tensor1.shape in
    let shape2 = tensor2.shape in

    let target_ndim = max (Array.length shape1) (Array.length shape2) in
    let padded_shape1 = pad_shape_left target_ndim shape1 in
    let padded_shape2 = pad_shape_left target_ndim shape2 in
    let padded_strides1 = compute_strides padded_shape1 in
    let padded_strides2 = compute_strides padded_shape2 in

    let out_shape = broadcast_shape padded_shape1 padded_shape2 in
    let result = create out_shape in
    let n = Array.length out_shape in
    let multi_idx = Array.make n 0 in
    let total = Array.fold_left ( * ) 1 out_shape in

    for flat_out = 0 to total - 1 do
      let fa = flat_idx padded_shape1 padded_strides1 multi_idx in
      let fb = flat_idx padded_shape2 padded_strides2 multi_idx in
      result.storage.{flat_out} <- op tensor1.storage.{fa} tensor2.storage.{fb};

      let j = ref (n - 1) in
      while !j >= 0 && (multi_idx.(!j) <- multi_idx.(!j) + 1; multi_idx.(!j) = out_shape.(!j)) do
        multi_idx.(!j) <- 0;
        j := !j - 1
      done
    done;
    result

  (* Initialization functions *)
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

  (* Operators *)
  let ( + ) t1 t2 = elementwise ( +. ) t1 t2
  let ( - ) t1 t2 = elementwise ( -. ) t1 t2
  let ( * ) t1 t2 = elementwise ( *. ) t1 t2

end
