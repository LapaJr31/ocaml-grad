module T = Tensor.Tensor

let print_shape shape =
  let shape_str =
    Array.to_list shape
    |> List.map string_of_int
    |> String.concat ", "
  in
  print_endline ("[" ^ shape_str ^ "]")

let print_tensor tensor =
  let shape = T.shape tensor in
  let total_len = T.len tensor in
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

    print_float tensor.T.storage.{i};

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

let () =
  let t1 = T.zeros [|2; 3|] in
  print_string "Tensor 1 shape: ";
  print_shape (T.shape t1);

  let data = [|1.0; 2.0; 3.0; 4.0|] in
  let t2 = T.of_data [|4|] data in
  print_string "Tensor 2 shape: ";
  print_shape (T.shape t2);

  let tensor = T.of_data [|2; 3|] [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
  print_tensor tensor;

  (* let random_tensor = T.random [|5;5;5|] (1.0, 5.0) 42 in *)
  (* print_tensor random_tensor; *)

  let t3 = T.random [|3; 1|] (1.0, 5.0) 42 in
  let t4 = T.random [|1; 4|] (1.0, 5.0) 42 in
  let sum = T.(t3 + t4) in
  print_tensor sum
