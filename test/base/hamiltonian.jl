# Test one-body terms

T = Float64
d = 10
N = 6

r = [ [ (k in (1,d+1) ? 1 : rand(1:6)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(Val(d),Val(N),r)

for i=1:d, j=1:d
  @test begin
    tij = randn()
    x = tij * AdagᵢAⱼ(x0, i, j)

    t = zeros(d,d); v = zeros(d,d,d,d);
    t[i,j] = tij; sH = SparseHamiltonian(t,v,Val(N),Val(d));
    y = sH * x0
    z = x-y

    norm(z, :LR) < norm(x, :LR)*1e-12
  end
end

# Test two-body terms

T = Float64
d = 6
N = 2

r = [ [ (k in (1,d+1) ? 1 : rand(1:6)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(Val(d),Val(N),r)

for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d #j=((1:i-1) ∪ (i+1:d)), k=1:d-1, l=((1:k-1) ∪ (k+1:d))
  @test begin
    tijkl = exp(randn())
    x = ((i<j ? 1 : -1) * (k<l ? 1 : -1) * tijkl) * AdagᵢAdagⱼAₖAₗ(x0, min(i,j), max(i,j), min(k,l), max(k,l))
    t = zeros(d,d); v = zeros(d,d,d,d);
    v[i,j,k,l] = tijkl; sH = SparseHamiltonian(t,v,Val(N),Val(d));
    y = sH * x0
    z = x-y

    norm(z, :LR) < norm(x, :LR)*1e-12
  end
end

T = Float64
d = 6
N = 3

using Graphs, MetaGraphsNext, GraphRecipes, Plots
for i=1:d-1,j=i+1:d
  @test begin
    t = zeros(d,d); v = zeros(d,d,d,d)
    t[i,j] = 2.0
    H = SparseHamiltonian(t,v,Val(N),Val(d))
    states, g = H.states, H.graph

    n = nv(g.graph)
    x = zeros(n); y = zeros(n); z = zeros(n)

    for i=1:n
      k, m, s, idx = g.vertex_labels[i]
      x[i] = k + .75 * ( (idx[1]-1)/(d-1) -1/2)
      y[i] = m+.25*s[1]
    end

    # display( (i,j) )
    # graphplot(g.graph, x=x, y=y, curves=false)

    u = findfirst(isequal((1,0,(0,0,1),(0,0))), g.vertex_labels)
    v = findfirst(isequal((d+1,N,(0,1,0),(0,0))), g.vertex_labels)
    u != nothing && v!= nothing && has_path(g.graph, u, v)
  end
end

for i=1:d-1,j=i+1:d,k=1:d-1,l=k+1:d
  @test begin
    t = zeros(d,d); v = zeros(d,d,d,d)
    v[i,j,k,l] = 2.0
    H = SparseHamiltonian(t,v,Val(N),Val(d))
    states, g = H.states, H.graph

    n = nv(g.graph)
    x = zeros(n); y = zeros(n); z = zeros(n)

    for i=1:n
      k, s, idx = g.vertex_labels[i]
      if length(idx) == 1
        x[i] = k
        y[i] = s[1] + .75 * ( (idx-1)/(d-1) -.5)
      else
        x[i] = k + .75 * ( (idx[1]-1)/(d-1) -.5)
        y[i] = s[1] + .75 * ( (idx[2]-1)/(d-1) -.5)
      end
    end

    # display( (i,j,k,l) )
    # graphplot(g.graph, x=x, y=y, curves=false)

    u = findfirst(isequal((1,0,(0,0,2),(0,0))), g.vertex_labels)
    v = findfirst(isequal((d+1,N,(0,2,0),(0,0))), g.vertex_labels)
    u != nothing && v!= nothing && has_path(g.graph, u, v)
  end
end

# Test one-body terms

T = Float64
d = 10
N = 6

r = [ [ (k in (1,d+1) ? 1 : rand(1:6)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(Val(d),Val(N),r)

for i=1:d, j=1:d
  @test begin
    tij = randn()
    x = tij * AdagᵢAⱼ(x0, i, j)

    t = zeros(d,d); v = zeros(d,d,d,d);
    t[i,j] = tij; 
    sH = SparseHamiltonian(t,v,Val(N),Val(d))
    y = sH * x0
    z = x-y

    norm(z, :LR) < norm(x, :LR)*1e-12
  end
end

# Test two-body terms

T = Float64
d = 6
N = 2

r = [ [ (k in (1,d+1) ? 1 : rand(1:6)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]
x0 = tt_randn(Val(d),Val(N),r)

for i=1:d-1, j=i+1:d, k=1:d-1, l=k+1:d #j=((1:i-1) ∪ (i+1:d)), k=1:d-1, l=((1:k-1) ∪ (k+1:d))
  @test begin
    tijkl = exp(randn())
    x = ((i<j ? 1 : -1) * (k<l ? 1 : -1) * tijkl) * AdagᵢAdagⱼAₖAₗ(x0, min(i,j), max(i,j), min(k,l), max(k,l))
    t = zeros(d,d); v = zeros(d,d,d,d);
    v[i,j,k,l] = tijkl;
    sH = SparseHamiltonian(t,v,Val(N),Val(d))
    y = sH * x0
    z = x-y

    norm(z, :LR) < norm(x, :LR)*1e-12
  end
end









