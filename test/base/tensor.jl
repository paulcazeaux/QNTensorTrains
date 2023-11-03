#tt_state
@test begin
  T = Float64
  d = 10
  N = 6

  s = [2,4,6,7,8,10]
  x = tt_state([k∈s for k=1:d])
  X = Array(x)
  I0 = CartesianIndex(Tuple( (i in s ? 2 : 1) for i=1:d ))

  pass=true
  for I in CartesianIndices(X)
    if !(((I == I0) && (X[I]==T(1))) || (X[I] == T(0)))
      @show I0, X[I0], I, X[I], X[I0]==T(1), X[I]==T(0)
      pass=false
      break
    end
  end

  pass
end

#tt_zeros
@test begin
  T = Float64
  d = 10
  N = 4

  x = tt_zeros(d,N)
  X = Array(x)
  
  pass=true
  for I in CartesianIndices(X)
    if !(X[I] == T(0))
      pass=false
      break
    end
  end

  pass
end

#tt_ones
@test begin
  T = Float64
  d = 10
  N = 5

  x = tt_ones(d,N)
  X = Array(x)
  
  pass=true
  for I in CartesianIndices(X)
    n = sum(Tuple(I).==2)
    if !( n==N && X[I]==T(1) || (X[I]==T(0)) )
      pass=false
      break
    end
  end

  pass
end


# tt_randn
@test begin
  T = Float64
  d = 10
  N = 6

  r = [ [ (k in (1,d+1) ? 1 : rand(0:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  x = tt_randn(Val(d),Val(N),r)
  X = Array(x)
  
  pass=true
  for I in CartesianIndices(X)
    n = sum(Tuple(I).==2)
    if !( n==N && X[I]!=T(0) || (X[I]==T(0)) )
      pass=false
      break
    end
  end

  pass
end

# add_non_essential_dims
@test begin
  T = Float64
  d = 5
  N = 3

  s = [1,2,4]
  x = tt_state([k∈s for k=1:d])

  new_d = 10
  old_dims = (2,3,4,6,10)
  y = QNTensorTrains.add_non_essential_dims(x, new_d, old_dims)
  yn = tt_state([k∈old_dims[s] for k=1:new_d])

  norm(Array(y) - Array(yn)) < eps()
end

# add
@test begin
  T = Float64
  d = 10
  N = 6

  r1 = [ [ (k in (1,d+1) ? 1 : rand(1:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  r2 = [ [ (k in (1,d+1) ? 1 : rand(1:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x1 = tt_randn(Val(d),Val(N),r1)
  x2 = tt_randn(Val(d),Val(N),r2)
  X1 = Array(x1)
  X2 = Array(x2)
  α = randn()
  β = randn()

  norm( Array(α*x1+β*x2) - (α*X1+β*X2) ) / norm(α*X1+β*X2) < 2d*eps()
end

# substract
@test begin
  T = Float64
  d = 10
  N = 6

  r1 = [ [ (k==1 || k==d+1 ? 1 : rand(1:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  r2 = [ [ (k==1 || k==d+1 ? 1 : rand(1:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x1 = tt_randn(Val(d),Val(N),r1)
  x2 = tt_randn(Val(d),Val(N),r2)
  X1 = Array(x1)
  X2 = Array(x2)
  α = randn()
  β = randn()

  norm( Array(α*x1-β*x2) - (α*X1-β*X2) ) / norm(α*X1-β*X2) < 2d*eps()
end

# Entrywise scalar product
@test begin
  T = Float64
  d = 10
  N = 6

  r = [ [ (k==1 || k==d+1 ? 1 : rand(1:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x = tt_randn(Val(d),Val(N),r)
  c = pi
  X = Array(x)

  norm( Array(c*x) - (c*X) ) / norm(c*X) + norm( Array(x*c) - (c*X) ) / norm(c*X) < 22d*eps()
end


# Hadamard product
@test begin
  T = Float64
  d = 10
  N = 6

  r1 = [ [ (k==1 || k==d+1 ? 1 : rand(1:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  r2 = [ [ (k==1 || k==d+1 ? 1 : rand(1:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
  
  x1 = tt_randn(Val(d),Val(N),r1)
  x2 = tt_randn(Val(d),Val(N),r2)
  X1 = Array(x1)
  X2 = Array(x2)

  norm( Array(times(x1,x2)) - (X1.*X2) ) / norm(X1.*X2) < 2d*eps()
end
