// Toy DCE pass placeholder test.
// Expected behavior after implementation: remove toy.mul because it has no users.
// module {
//   func.func @dead_example(%arg0: tensor<?xf64>) -> tensor<?xf64> {
//     %0 = "toy.add"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
//     return %0 : tensor<?xf64>
//   }
// }

func.func @dead_example(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %sum = "toy.add"(%arg0, %arg0) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  %unused = "toy.mul"(%sum, %sum) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return %sum : tensor<?xf64>
}
