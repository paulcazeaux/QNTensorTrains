# Test with one-body terms

T = Float64
d = 9
N = 5

r = [ [ (k in (1,d+1) ? 1 : rand(2:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(Val(d),Val(N),r)

# Classical Lanczos
for i=1:d, j=1:d
  @test begin
    t = zeros(d,d); v = zeros(d,d,d,d);
    t[i,j] = t[j,i] = -1
    H = SparseHamiltonian(t,v,Val(N),Val(d))

    λ,x,breakdown = Lanczos(H,x0,tol=1e-6)
    if breakdown
      λ,x,breakdown = Lanczos(H,x,tol=1e-6)
      if breakdown
        λ,x,breakdown = randLanczos(H,x,tol=1e-6)
      end
    end

    # @show i,j,RayleighQuotient(H, x)
    isapprox(RayleighQuotient(H, x), -1; atol=1e-6) && isapprox(norm(H*x - λ*x, :LR), 0; atol=1e-6)
  end
end

# Randomized Lanczos
for i=1:d, j=1:d
  @test begin
    t = zeros(d,d); v = zeros(d,d,d,d);
    t[i,j] = t[j,i] = -1
    H = SparseHamiltonian(t,v,Val(N),Val(d))

    λ,x,breakdown = randLanczos(H,x0,tol=1e-6)
    if breakdown
      λ,x,breakdown = randLanczos(H,x,tol=1e-6)
      if breakdown
        λ,x,breakdown = randLanczos(H,x,tol=1e-6)
      end
    end

    # @show RayleighQuotient(H, x), norm(H*x - λ*x, :LR)
    isapprox(RayleighQuotient(H, x), -1; atol=1e-6) && isapprox(norm(H*x - λ*x, :LR), 0; atol=1e-6)
  end
end

# Test two-body terms

T = Float64
d = 6
N = 2

r = [ [ (k in (1,d+1) ? 1 : rand(5:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(Val(d),Val(N),r)

# Classical Lanczos
for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d
  @test begin
    t = zeros(T,d,d); v = zeros(T,d,d,d,d);
    v[i,j,k,l] = v[l,k,j,i] = -1;
    H = SparseHamiltonian(t,v,Val(N),Val(d))

    λ,x,breakdown = Lanczos(H,x0,tol=1e-6)
    if breakdown
      λ,x,breakdown = Lanczos(H,x,tol=1e-6)
      if breakdown
        λ,x,breakdown = randLanczos(H,x,tol=1e-6)
      end
    end
    z = tt_state([n∈(i,j) for n=1:d]) + tt_state([n∈(k,l) for n=1:d])
    round_global!(z)
    z *= 1/norm(z)
    
    # @show i,j,k,l,abs(dot(x,z))
    # @show norm(H*x - λ*x, :LR)
    isapprox(abs(dot(x,z)), 1.; atol=1e-6) && isapprox(norm(H*x - λ*x, :LR), 0; atol=1e-6)
  end
end

# Randomized Lanczos
for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d
  @test begin
    t = zeros(T,d,d); v = zeros(T,d,d,d,d);
    v[i,j,k,l] = v[l,k,j,i] = -1;
    H = SparseHamiltonian(t,v,Val(N),Val(d))

    λ,x,breakdown = randLanczos(H,x0,tol=1e-6)
    if breakdown
      λ,x,breakdown = randLanczos(H,x,tol=1e-6)
      if breakdown
        λ,x,breakdown = randLanczos(H,x,tol=1e-6)
      end
    end
    z = tt_state([n∈(i,j) for n=1:d]) + tt_state([n∈(k,l) for n=1:d])
    round_global!(z)
    z *= 1/norm(z)
    
    # @show abs(dot(x,z))
    # @show norm(H*x - λ*x, :LR)
    isapprox(abs(dot(x,z)), 1.; atol=1e-6) && isapprox(norm(H*x - λ*x, :LR), 0; atol=1e-6)
  end
end
