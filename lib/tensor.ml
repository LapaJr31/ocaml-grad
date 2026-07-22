open Bigarray

exception Shape_mismatch

type t = {
  shape : int array;
  strides : int array;
  offset: int;
  storage : (float, float32_elt, c_layout) Bigarray.Array1.t;
}


let len t = Bigarray.Array1.dim t.storage

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


let pad_strides_left target_ndim strides =
  let current_ndim = Array.length strides in
  let diff = target_ndim - current_ndim in
  if diff <= 0 then strides
  else
    Array.append (Array.make diff 0) strides

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

  let s1 = pad_strides_left target_ndim tensor1.strides in
  let s2 = pad_strides_left target_ndim tensor2.strides in

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

let contiguous t =
  let total = Array.fold_left ( * ) 1 t.shape in
  let dst = create t.shape in
  let n = Array.length t.shape in
  let multi_idx = Array.make n 0 in
  for flat = 0 to total - 1 do
    let src = flat_idx t multi_idx in
    dst.storage.{flat} <- t.storage.{src};
    let j = ref (n - 1) in
    while !j >= 0 && (multi_idx.(!j) <- multi_idx.(!j) + 1; multi_idx.(!j) = t.shape.(!j)) do
      multi_idx.(!j) <- 0;
      j := !j - 1
    done
  done;
  dst

let is_contiguous t =
  t.offset = 0
  && t.strides = compute_strides t.shape
  && len t = Array.fold_left ( * ) 1 t.shape

let ensure_contiguous t = if is_contiguous t then t else contiguous t

let shape t = t.shape

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

let get t coords = t.storage.{flat_idx t coords}

let to_array t =
  let total = Array.fold_left ( * ) 1 t.shape in
  let result = Array.make total 0.0 in
  let n = Array.length t.shape in
  let multi_idx = Array.make n 0 in
  for flat = 0 to total - 1 do
    result.(flat) <- t.storage.{flat_idx t multi_idx};
    let j = ref (n - 1) in
    while !j >= 0 && (multi_idx.(!j) <- multi_idx.(!j) + 1; multi_idx.(!j) = t.shape.(!j)) do
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
  let open Stdlib in
  let sa = t1.shape and sb = t2.shape in
  let na = Array.length sa and nb = Array.length sb in
  if na < 2 || nb < 2 then raise Shape_mismatch;
  let m = sa.(na-2) and k = sa.(na-1) in
  if k <> sb.(nb-2) then raise Shape_mismatch;
  let n = sb.(nb-1) in

  let batch_a = Array.sub sa 0 (na-2) and batch_b = Array.sub sb 0 (nb-2) in
  let nd = max (Array.length batch_a) (Array.length batch_b) in
  let pba = pad_shape_left nd batch_a and pbb = pad_shape_left nd batch_b in
  let out_batch = broadcast_shape pba pbb in
  let out_shape = Array.append out_batch [| m; n |] in

  let t1 = ensure_contiguous t1 and t2 = ensure_contiguous t2 in
  let sa_batch = broadcast_strides out_batch pba
                   (pad_strides_left nd (Array.sub t1.strides 0 (na-2))) in
  let sb_batch = broadcast_strides out_batch pbb
                   (pad_strides_left nd (Array.sub t2.strides 0 (nb-2))) in

  let result = zeros out_shape in
  let total = Array.fold_left ( * ) 1 out_batch in
  let idx = Array.make nd 0 in
  for c = 0 to total - 1 do
    let aoff = ref 0 and boff = ref 0 in
    for d = 0 to nd - 1 do
      aoff := !aoff + idx.(d) * sa_batch.(d);
      boff := !boff + idx.(d) * sb_batch.(d)
    done;
    Blas.cblas_gemm m n k
      (Array1.sub t1.storage !aoff (m*k))
      (Array1.sub t2.storage !boff (k*n))
      (Array1.sub result.storage (c*m*n) (m*n));
    let j = ref (nd-1) in
    while !j >= 0 && (idx.(!j) <- idx.(!j)+1; idx.(!j) = out_batch.(!j)) do
      idx.(!j) <- 0; j := !j - 1
    done
  done;
  result

(* Printing *)
let print_shape shape =
  let shape_str =
    Array.to_list shape
    |> List.map string_of_int
    |> String.concat ", "
  in
  print_endline ("[" ^ shape_str ^ "]")

let print_tensor tensor =
  let open Stdlib in
  let data = to_array tensor in
  let shape = shape tensor in
  let total_len = Array.fold_left ( * ) 1 shape in
  let num_dims = Array.length shape in

  if num_dims = 0 then print_endline "[]" else

  for i = 0 to total_len - 1 do
    let coords = Array.make num_dims 0 in
    let temp = ref i in
    for d = num_dims - 1 downto 0 do
      coords.(d) <- !temp mod shape.(d);
      temp := !temp / shape.(d)
    done;

    for d = 0 to num_dims - 1 do
      let starts_here = ref true in
      for k = d to num_dims - 1 do
        if coords.(k) <> 0 then starts_here := false
      done;
      if !starts_here then print_string "["
    done;

    print_float data.(i);

    let trailing_closes = ref 0 in
    for d = num_dims - 1 downto 0 do
      let ends_here = ref true in
      for k = d to num_dims - 1 do
        if coords.(k) <> shape.(k) - 1 then ends_here := false
      done;
      if !ends_here then incr trailing_closes
    done;

    if !trailing_closes > 0 then begin
      for _ = 1 to !trailing_closes do print_string "]" done;
      if i < total_len - 1 then print_string ",\n "
    end else
      print_string ", "
  done;
  print_newline ()
