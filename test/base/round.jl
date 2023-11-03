# Left orthogonalize
@test begin
  T = Float64
  d = 10
  N = 3

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r)
  X = Array(x)
  QNTensorTrains.leftOrthogonalize!(x)

  isapprox( norm( X - Array(x) ), 0, atol = 1e-12 * norm(X) )
end

# Left orthogonalize - rank conserving
@test begin
  T = Float64
  d = 10
  N = 3

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r)
  X = Array(x)
  QNTensorTrains.leftOrthogonalize!(x, keepRank=true)

  isapprox( norm( X - Array(x) ), 0.0,  atol=1e-12 * norm(X) ) && all(all(rank(x,k) .== r[k]) for k=1:d+1)
end

# Right orthogonalize
@test begin
  T = Float64
  d = 10
  N = 3

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r)
  X = Array(x)
  QNTensorTrains.rightOrthogonalize!(x)

  isapprox( norm( X - Array(x) ), 0, atol = 1e-12 * norm(X) )
end

# Right orthogonalize - rank conserving
@test begin
  T = Float64
  d = 10
  N = 3

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r)
  X = Array(x)
  QNTensorTrains.rightOrthogonalize!(x, keepRank=true)

  isapprox( norm( X - Array(x) ), 0.0, atol = 1e-12 * norm(X) ) && all(all(rank(x,k) .== r[k]) for k=1:d+1)
end

# Round
@test begin
  T = Float64
  d = 10
  N = 7
  tol = 1e-8

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r)
  X = Array(x)
  y = round(x, tol)
  Y = Array(y)

  isapprox( norm( X-Y ), 0, atol = tol * norm(X) )
end


# Global round
@test begin
  T = Float64
  d = 10
  N = 7
  tol = 1e-8

  r = [ [ (k==1 || k==d+1 ? 1 : rand(0:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r)
  X = Array(x)
  y = round(x, tol; Global=true)
  Y = Array(y)

  isapprox( norm( X-Y ), 0, atol = tol * norm(X) )
end
