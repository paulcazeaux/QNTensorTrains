
# dot
@test begin
  T = Float64
  d = 10
  N = 5
  Sz = 1//2
  
  x = randomized_state(d, N, Sz, 1:5)
  y = randomized_state(d, N, Sz, 1:5)
  X = Array(x)
  Y = Array(y)

  isapprox( dot(x,y), sum(X.*Y); rtol = 1e-8)
end


# dot with orthogonalization
@test begin
  T = Float64
  d = 10
  N = 5
  Sz = -1//2

  x = randomized_state(d, N, Sz, 5:10)
  y = randomized_state(d, N, Sz, 5:10)
  X = Array(x)
  Y = Array(y)
  
  isapprox(sum(X.*Y), dot(x,y,orthogonalize=true); rtol = 1e-8)
end