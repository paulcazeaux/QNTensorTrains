
# dot
@test begin
  T = Float64
  d = 10
  N = 5
  Sz = 1//2

  Nup = Int(N+2Sz)÷2
  Ndn = Int(N-2Sz)÷2
  r1 = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  r2 = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
    r1[k][nup,ndn] = (k in (1,d+1) ? 1 : rand(1:5))
    r2[k][nup,ndn] = (k in (1,d+1) ? 1 : rand(1:5))
  end
  
  x = tt_randn(Val(d),Val(Nup),Val(Ndn),r1)
  y = tt_randn(Val(d),Val(Nup),Val(Ndn),r2)
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

  Nup = Int(N+2Sz)÷2
  Ndn = Int(N-2Sz)÷2
  r1 = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  r2 = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,k)
    r1[k][nup,ndn] = (k in (1,d+1) ? 1 : rand(5:10))
    r2[k][nup,ndn] = (k in (1,d+1) ? 1 : rand(5:10))
  end

  x = tt_randn(Val(d),Val(Nup),Val(Ndn),r1)
  y = tt_randn(Val(d),Val(Nup),Val(Ndn),r2)
  X = Array(x)
  Y = Array(y)
  
  isapprox(sum(X.*Y), dot(x,y,orthogonalize=true); rtol = 1e-8)
end