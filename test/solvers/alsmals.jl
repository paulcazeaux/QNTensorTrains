# Test with one-body terms

T = Float64
d = 9
N = 2
Sz = 2//2


for i=1:d, j=1:d
  @test begin
    t = zeros(d,d); v = zeros(d,d,d,d);
    t[i,j] = t[j,i] = -1

    H = SparseHamiltonian(t,v,N,Sz,d)
    x0 = randomized_state(d, N, Sz, 5:10)
    μ,y = MALS(H,x0,1e-9)
    λ,x = ALS(H,x0,1e-9)

    RayleighQuotient(H,x) ≈ -1 && isapprox(norm(H*x - λ*x, :LR), 0; atol=1e-6) &&
    RayleighQuotient(H,y) ≈ -1 && isapprox(norm(H*y - μ*y, :LR), 0; atol=1e-6)
  end
end

# Test two-body terms

T = Float64
d = 6
N = 4
Sz = 0//2


for i=1:d, j=filter(≠(i), 1:d), k=filter(∉((i,j)), 1:d), l=filter(∉((i,j,k)), 1:d)
  @test begin
    t = zeros(T,d,d); v = zeros(T,d,d,d,d);
    v[i,j,k,l] = v[l,k,j,i] = -1;

    H = SparseHamiltonian(t,v,N,Sz,d)

    z = tt_state([n∈(i,k) for n=1:d], [n∈(i,k) for n=1:d]) + tt_state([n∈(l,j) for n=1:d], [n∈(l,j) for n=1:d])
    round_global!(z)
    z *= 1/norm(z)
    λ = RayleighQuotient(H,z)

    x0 = randomized_state(d, N, Sz, 5:10)
    λ₁,x₁ = ALS(H,x0,1e-6)
    x0 = randomized_state(d, N, Sz, 5:10)
    λ₂,x₂ = MALS(H,x0,1e-6)

    for i=1:9
      λ₁≈λ && break
      x0 = randomized_state(d, N, Sz, 5:10)
      λ₁,x₁ = ALS(H,x0,1e-6)
    end
    for i=1:9
      λ₂≈λ && break
      x0 = randomized_state(d, N, Sz, 5:10)
      λ₂,x₂ = MALS(H,x0,1e-6)
    end

    @show i,j,k,l, λ, λ₁, dot(x₁,z), λ₂, dot(x₂,z)
    isapprox(abs(dot(x₁,z)), 1.; atol=1e-6) && isapprox(norm(H*x₁ - λ₁*x₁, :LR), 0; atol=1e-6) &&
    isapprox(abs(dot(x₂,z)), 1.; atol=1e-6) && isapprox(norm(H*x₂ - λ₂*x₂, :LR), 0; atol=1e-6)
  end
end
