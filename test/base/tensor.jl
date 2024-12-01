#tt_state
@test begin
  T = Float64
  d = 10
  N = 6
  Sz = 0//2

  s1 = [2,4,6]
  s2 = [7,8,10]
  x = tt_state([k∈s1 for k=1:d], [k∈s2 for k=1:d])
  X = Array(x)
  I0 = CartesianIndex(Tuple( (i in s1 && i in s2 ? 4 : i in s2 ? 3 : i in s1 ? 2 : 1) for i=1:d ))

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
  Sz = -2//2

  x = tt_zeros(d,N,Sz)
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
  Sz = 1//2

  x = tt_ones(d,N,Sz)
  X = Array(x)
  
  pass=true
  for I in CartesianIndices(X)
    n = sum(Tuple(I).==2)+sum(Tuple(I).==3)+2*sum(Tuple(I).==4)
    sz = (sum(Tuple(I).==2)-sum(Tuple(I).==3))//2
    if !( X[I] == (n==N && sz==Sz ? T(1) : T(0) ) )
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
  Sz= 2//2

  x = randomized_state(d,N,Sz,1:5)
  X = Array(x)
  
  pass=true
  for I in CartesianIndices(X)
    n = sum(Tuple(I).==2)+sum(Tuple(I).==3)+2*sum(Tuple(I).==4)
    sz = (sum(Tuple(I).==2)-sum(Tuple(I).==3))//2
    if !(  (n==N && sz==Sz ? X[I] != T(0) : X[I]==T(0) ) )
      @show I, n, sz, N, Sz
      @show X[I]
      lup, ldn = 1, 1
      for k=1:d
        if I[k] == 1
          rup,rdn = lup, ldn
        elseif I[k] == 2
          rup,rdn = lup+1, ldn
        elseif I[k] == 3
          rup,rdn = lup, ldn+1
        elseif I[k] == 4
          rup,rdn = lup+1, ldn+1
        end
        @show k, I[k], core(x,k)[(lup,ldn),(rup,rdn)]
        lup,ldn = rup,rdn
      end

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
  N = 4
  Sz = 0//2

  sup = [1,2]
  sdn = [2,4]
  x = tt_state([k∈sup for k=1:d],[k∈sdn for k=1:d])

  new_d = 10
  old_dims = (2,3,4,6,10)
  y = QNTensorTrains.add_non_essential_dims(x, new_d, old_dims)
  yn = tt_state([k∈old_dims[sup] for k=1:new_d],[k∈old_dims[sdn] for k=1:new_d])

  norm(Array(y) - Array(yn)) < eps()
end

# add
@test begin
  T = Float64
  d = 10
  N = 6
  Sz = 2//2
  
  x1 = randomized_state(d,N,Sz,1:5)
  x2 = randomized_state(d,N,Sz,1:5)
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
  Sz = -2//2
  
  x1 = randomized_state(d,N,Sz,1:5)
  x2 = randomized_state(d,N,Sz,1:5)
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
  Sz = 4//2

  x = randomized_state(d,N,Sz,1:5)
  c = pi
  X = Array(x)

  norm( Array(c*x) - (c*X) ) / norm(c*X) + norm( Array(x*c) - (c*X) ) / norm(c*X) < 22d*eps()
end


# Hadamard product
@test begin
  T = Float64
  d = 10
  N = 7
  Sz = -1//2
  
  x1 = randomized_state(d,N,Sz,1:5)
  x2 = randomized_state(d,N,Sz,1:5)
  X1 = Array(x1)
  X2 = Array(x2)

  norm( Array(times(x1,x2)) - (X1.*X2) ) / norm(X1.*X2) < 2d*eps()
end
