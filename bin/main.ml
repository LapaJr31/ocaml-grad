module T = Mentat.Tensor

let () =
  let t1 = T.zeros [|2; 3|] in
  print_string "Tensor 1 shape: ";
  T.print_shape (T.shape t1);

  let data = [|1.0; 2.0; 3.0; 4.0|] in
  let t2 = T.of_data [|4|] data in
  print_string "Tensor 2 shape: ";
  T.print_shape (T.shape t2);

  let tensor = T.of_data [|2; 3|] [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
  T.print_tensor tensor;

  (* let random_tensor = T.random [|5;5;5|] (1.0, 5.0) 42 in *)
  (* T.print_tensor random_tensor; *)

  let t3 = T.random [|4; 4; 4|] (1.0, 5.0) ~seed:42 () in
  let t4 = T.random [|4; 4; 4|] (1.0, 5.0) ~seed:99 () in

  print_string "\n Addition \n";
  let sum = T.(t3 + t4) in
  T.print_tensor sum;

  print_string "\n Subtraction \n";
  let sub = T.(t3 - t4) in
  T.print_tensor sub;

  print_string "\n Multiplication \n";
  let mult = T.(t3 * t4) in
  T.print_tensor mult;

  print_string "\n Matmul \n";
  let res = T.matmul t3 t4 in
  T.print_tensor res;

  let a = T.of_data [|3;2|] [|1.;2.; 3.;4.; 5.;6.|] in
  let b = T.of_data [|2;4|] [|1.;1.;1.;0.; 1.;1.;0.;1.|] in
  T.print_tensor (T.matmul a b)
