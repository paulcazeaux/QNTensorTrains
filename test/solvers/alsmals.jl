# Test with one-body terms

T = Float64
d = 9
N = 5

r = [ [ (k in (1,d+1) ? 1 : rand(2:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(Val(d),Val(N),r)

for i=1:d, j=1:d
  @test begin
    t = zeros(d,d); v = zeros(d,d,d,d);
    t[i,j] = t[j,i] = -1

    H = SparseHamiltonian(t,v,Val(N),Val(d))
    λ,x = ALS(H,x0,1e-6)
    μ,y = MALS(H,x0,1e-6)

    RayleighQuotient(H,x) ≈ -1 && isapprox(norm(H*x - λ*x, :LR), 0; atol=1e-6) &&
    RayleighQuotient(H,y) ≈ -1 && isapprox(norm(H*y - μ*y, :LR), 0; atol=1e-6)
  end
end

# Test two-body terms

T = Float64
d = 6
N = 2

r = [ [ (k in (1,d+1) ? 1 : rand(5:10)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(Val(d),Val(N),r)

for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d
  @test begin
    t = zeros(T,d,d); v = zeros(T,d,d,d,d);
    v[i,j,k,l] = v[l,k,j,i] = -1;

    H = SparseHamiltonian(t,v,Val(N),Val(d))
    λ,x = ALS(H,x0,1e-6)
    μ,y = MALS(H,x0,1e-6)

    z = tt_state([n∈(i,j) for n=1:d]) + tt_state([n∈(k,l) for n=1:d])
    round_global!(z)
    z *= 1/norm(z)
    
    isapprox(abs(dot(x,z)), 1.; atol=1e-6) && isapprox(norm(H*x - λ*x, :LR), 0; atol=1e-6) &&
    isapprox(abs(dot(y,z)), 1.; atol=1e-6) && isapprox(norm(H*y - μ*y, :LR), 0; atol=1e-6)
  end
end
