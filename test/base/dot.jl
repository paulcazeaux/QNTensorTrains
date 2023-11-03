
# dot
@test begin
  T = Float64
  d = 10
  N = 5

  r1 = [ [ (k==1 || k==d+1 ? 1 : rand(0:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  r2 = [ [ (k==1 || k==d+1 ? 1 : rand(0:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r1)
  y = tt_randn(Val(d),Val(N),r2)
  X = Array(x)
  Y = Array(y)

  isapprox( dot(x,y), sum(X.*Y); rtol = 1e-8)
end


# dot with orthogonalization
@test begin
  T = Float64
  d = 10
  N = 5

  r1 = [ [ (k==1 || k==d+1 ? 1 : rand(5:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  r2 = [ [ (k==1 || k==d+1 ? 1 : rand(5:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r1)
  y = tt_randn(Val(d),Val(N),r2)
  X = Array(x)
  Y = Array(y)
  
  isapprox(sum(X.*Y), dot(x,y,orthogonalize=true); rtol = 1e-8)
end