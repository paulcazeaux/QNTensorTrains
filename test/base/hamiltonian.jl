# Test one-body terms

T = Float64
d = 10
N = 6

r = [ [ (k in (1,d+1) ? 1 : rand(1:6)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(d,N,r)

for i=1:d, j=1:d
  @test begin
    tij = randn()
    x = tij * AdagᵢAⱼ_view(x0, i, j)

    t = zeros(d,d); v = zeros(d,d,d,d);
    t[i,j] = tij; y = H_matvec(x0, t, v)
    z = x-y

    norm(z, :LR) < norm(x, :LR)*1e-12
  end
end

# Test two-body terms

T = Float64
d = 6
N = 2

r = [ [ (k in (1,d+1) ? 1 : rand(1:6)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(d,N,r)

for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d #j=((1:i-1) ∪ (i+1:d)), k=1:d-1, l=((1:k-1) ∪ (k+1:d))
  @test begin
    tijkl = exp(randn())
    x = ((i<j ? 1 : -1) * (k<l ? 1 : -1) * tijkl) * AdagᵢAdagⱼAₖAₗ(x0, min(i,j), max(i,j), min(k,l), max(k,l))
    t = zeros(d,d); v = zeros(d,d,d,d);
    v[i,j,k,l] = tijkl; y = H_matvec(x0, t, v)
    z = x-y

    norm(z, :LR) < norm(x, :LR)*1e-12
  end
end
