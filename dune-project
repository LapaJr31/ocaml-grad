(lang dune 3.20)

(name mentat)

(generate_opam_files true)

(source
 (github LapaJr31/mentat))

(authors "Yaroslav Latypov <jarlat04@lapuas.xyz>")

(license Apache-2.0)

(package
 (name mentat)
 (synopsis "A machine learning and linear algebra library in OCaml")
 (description "Mentat is an OCaml library providing n-dimensional tensors, broadcasting, and matrix operations.")
 (depends
  (ocaml (>= 4.14.0)))
 (tags
  ("machine-learning" "linear-algebra" "tensor" "dune" "ocaml")))
