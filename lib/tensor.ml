open Bigarray

exception Shape_mismatch

type t = {
  shape : int array;
  strides : int array;
  offset: int;
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
  let offset = 0 in
  { shape = input_shape; strides; offset; storage }

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

let broadcast_strides out_shape padded_shape real_strides =
  let n = Array.length out_shape in
  Array.init n (fun i ->
    if padded_shape.(i) = 1 then 0
    else real_strides.(i)
  )

let flat_idx t multi_idx =
  let acc = ref t.offset in
  for d = 0 to Array.length t.shape - 1 do
    acc := !acc + multi_idx.(d) * t.strides.(d)
  done;
  !acc

let elementwise op tensor1 tensor2 =
  let shape1 = tensor1.shape in
  let shape2 = tensor2.shape in

  let target_ndim = max (Array.length shape1) (Array.length shape2) in
  let padded_shape1 = pad_shape_left target_ndim shape1 in
  let padded_shape2 = pad_shape_left target_ndim shape2 in

  let out_shape = broadcast_shape padded_shape1 padded_shape2 in
  let result = create out_shape in
  let n = Array.length out_shape in
  let multi_idx = Array.make n 0 in
  let total = Array.fold_left ( * ) 1 out_shape in

  let s1 = compute_strides padded_shape1 in
  let s2 = compute_strides padded_shape2 in

  let t1_padded = {
    tensor1 with
    shape = padded_shape1;
    strides = broadcast_strides out_shape padded_shape1 s1
  } in
  let t2_padded = {
    tensor2 with
    shape = padded_shape2;
    strides = broadcast_strides out_shape padded_shape2 s2
  } in

  for flat_out = 0 to total - 1 do
    let fa = flat_idx t1_padded multi_idx in
    let fb = flat_idx t2_padded multi_idx in
    result.storage.{flat_out} <- op t1_padded.storage.{fa} t2_padded.storage.{fb};

    let j = ref (n - 1) in
    while !j >= 0 && (multi_idx.(!j) <- multi_idx.(!j) + 1; multi_idx.(!j) = out_shape.(!j)) do
      multi_idx.(!j) <- 0;
      j := !j - 1
    done
  done;
  result

let check_matmul_compatible t1 t2 =
  let shape_a = t1.shape in
  let shape_b = t2.shape in
  let dims_a = Array.length shape_a in
  let dims_b = Array.length shape_b in
  if dims_a < 2 || dims_b < 2 then false
  else begin
    let cols_a = shape_a.(dims_a - 1) in
    let rows_b = shape_b.(dims_b - 2) in
    cols_a = rows_b
  end

let slice t ranges =
  let ndim = Array.length t.shape in
  let new_shape = Array.copy t.shape in
  let new_offset = ref t.offset in
  for d = 0 to ndim - 1 do
    let (start, stop) = ranges.(d) in
    new_offset := !new_offset + start * t.strides.(d);
    new_shape.(d) <- stop - start
  done;
  { t with shape = new_shape; offset = !new_offset }

let get_row t i = slice t [| (i, i+1); (0, t.shape.(1)) |]
let get_col t j = slice t [| (0, t.shape.(0)); (j, j+1) |]

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

let random input_shape (x, y) ?seed () =
  (match seed with
  | Some s -> Random.init s
  | None   -> Random.self_init ());

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
(* THIS IS ELEMENTWISE MULTIPLICATION NOT MATMUL *)
let ( * ) t1 t2 = elementwise ( *. ) t1 t2

(* Now this is some proper matmul *)
let matmul t1 t2 =
  if not (check_matmul_compatible t1 t2) then
    raise Shape_mismatch
  else
  let ndim1 = Array.length t1.shape in
  let ndim2 = Array.length t2.shape in

  let ( - ) = Stdlib.( - ) in
  let m = t1.shape.(ndim1 - 2) in
  let k = t1.shape.(ndim1 - 1) in
  let n = t2.shape.(ndim2 - 1) in
  let result = zeros [| m; n |] in
  Blas.cblas_gemm m n k t1.storage t2.storage result.storage;
  result
