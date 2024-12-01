"""
  A = randn(Nup::Int, Ndn::Int, d::Int, k::Int,
                    row_ranks::Matrix{Int}, 
                    col_ranks::Matrix{Int},
                    [T=Float64])

  Compute a core with randomized entries within the sparse block structure allowed
  for the `k`-th core of a `d`-dimensional TT-tensor with `Nup`/`Ndn` occupation numbers,
  with given ranks `row_ranks` and `col_ranks`.
"""
function Base.randn(Nup::Int, Ndn::Int, d::Int, k::Int,
                    row_ranks::Matrix{Int}, 
                    col_ranks::Matrix{Int},
                    ::Type{T}=Float64; 
                    orthogonal = :none) where T<:Number
  A = SparseCore{T,Nup,Ndn,d}(k, row_ranks, col_ranks)
  row_qn = state_qn(Nup,Ndn,d,k)
  col_qn = state_qn(Nup,Ndn,d,k+1)
  for (nup,ndn) in it_○○(A)
    if row_ranks[nup,ndn]>0 && col_ranks[nup,ndn]>0
      σ = ( row_ranks[nup,ndn]*col_ranks[nup,ndn] )^(-1/2)
      randn!(○○(A,nup,ndn))
      ○○(A,nup,ndn) .*= σ
    end
  end
  for (nup,ndn) in it_up(A)
    if row_ranks[nup,ndn]>0 && col_ranks[nup+1,ndn]>0
      σ = ( row_ranks[nup,ndn]*col_ranks[nup+1,ndn] )^(-1/2)
      randn!(up(A,nup,ndn))
      up(A,nup,ndn) .*= σ
    end
  end
  for (nup,ndn) in it_dn(A)
    if row_ranks[nup,ndn]>0 && col_ranks[nup,ndn+1]>0
      σ = ( row_ranks[nup,ndn]*col_ranks[nup,ndn+1] )^(-1/2)
      randn!(dn(A,nup,ndn))
      dn(A,nup,ndn) .*= σ
    end
  end
  for (nup,ndn) in it_●●(A)
    if row_ranks[nup,ndn]>0 && col_ranks[nup+1,ndn+1]>0
      σ = ( row_ranks[nup,ndn]*col_ranks[nup+1,ndn+1] )^(-1/2)
      randn!(●●(A,nup,ndn))
      ●●(A,nup,ndn) .*= σ
    end
  end

  if orthogonal==:left
    qr!(A)
  elseif orthogonal==:right
    lq!(A)
  end
  return A
end

"""
  A = rand(S, Nup::Int, Ndn::Int, d::Int, k::Int,
                    row_ranks::Matrix{Int}, 
                    col_ranks::Matrix{Int})

  Compute a core with randomized entries in the indexable collection `S` within the sparse block structure allowed
  for the `k`-th core of a `d`-dimensional TT-tensor with `N` total occupation number,
  with given ranks `row_ranks` and `col_ranks`.
"""
function Base.rand(S, Nup::Int, Ndn::Int, d::Int, k::Int,
                    row_ranks::Matrix{Int}, 
                    col_ranks::Matrix{Int},
                    ::Type{T}=Float64; 
                    orthogonal = :none) where {T<:Number}

  @assert eltype(S) <: Number
  @assert T == float(eltype(S))

  A = SparseCore{T,Nup,Ndn,d}(k, row_ranks, col_ranks)

  function randomize!(A)
    for (nup,ndn) in it_○○(A)
      if row_ranks[nup,ndn]>0 && col_ranks[nup,ndn]>0
        a = float(rand(S,row_ranks[nup,ndn],col_ranks[nup  ,ndn  ]))
        copyto!(○○(A,nup,ndn), a)
      end
    end
    for (nup,ndn) in it_up(A)
      if row_ranks[nup,ndn]>0 && col_ranks[nup+1,ndn]>0
        a = float(rand(S,row_ranks[nup,ndn],col_ranks[nup+1,ndn  ]))
        copyto!(up(A,nup,ndn), a)
      end
    end
    for (nup,ndn) in it_dn(A)
      if row_ranks[nup,ndn]>0 && col_ranks[nup,ndn+1]>0
        a = float(rand(S,row_ranks[nup,ndn],col_ranks[nup  ,ndn+1]))
        copyto!(dn(A,nup,ndn), a)
      end
    end
    for (nup,ndn) in it_●●(A)
      if row_ranks[nup,ndn]>0 && col_ranks[nup,ndn]>0
        a = float(rand(S,row_ranks[nup,ndn],col_ranks[nup+1,ndn+1]))
        copyto!(●●(A,nup,ndn), a)
      end
    end
  end

  while true
    randomize!(a)
    for nup in row_qn(A)
      a = A[(nup,ndn),:vertical]
      r = minimum(size(a))
      r<50 && rank(a) < r && continue
    end
    for nup in col_qn(A)
      a = A[(nup,ndn),:horizontal]
      r = minimum(size(a))
      r<50 && rank(a) < r && continue
    end
    break
  end
  
  if orthogonal==:left
    qr!(A)
  elseif orthogonal==:right
    lq!(A)
  end
  return A
end

"""
  A = randd(Val(d), Val(Nup), Val(Ndn), k::Int, r::Int,
                    [T=Float64])

  Compute a 'diagonal' core i.e. A[:,1,:] and A[:,2,:] are diagonal and Gaussian distributed for 1<k<d.
"""
function randd(::Val{d}, ::Val{Nup}, ::Val{Ndn}, k::Int, r::Int, ::Type{T}=Float64) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert 1<k<d
  ω₁ = Diagonal(randn(T,r))
  ω₂ = Diagonal(randn(T,r))
  ω₃ = Diagonal(randn(T,r))
  ω₄ = Diagonal(randn(T,r))

  row_qn = state_qn(Nup,Ndn,d,k)
  col_qn = state_qn(Nup,Ndn,d,k+1)

  ○○ = Matrix{Diagonal{T,Vector{T}}}(undef, Nup+1, Ndn+1)
  up = Matrix{Diagonal{T,Vector{T}}}(undef, Nup+1, Ndn+1)
  dn = Matrix{Diagonal{T,Vector{T}}}(undef, Nup+1, Ndn+1)
  ●● = Matrix{Diagonal{T,Vector{T}}}(undef, Nup+1, Ndn+1)

  for (nup,ndn) in row_qn
    (nup  ,ndn  ) in col_qn && (○○[nup,ndn] = ω₁)
    (nup+1,ndn  ) in col_qn && (up[nup,ndn] = ω₂)
    (nup  ,ndn+1) in col_qn && (dn[nup,ndn] = ω₃)
    (nup+1,ndn+1) in col_qn && (●●[nup,ndn] = ω₄)
  end

  Ωₖ = SparseCore{T,Nup,Ndn,d}(k, ○○, up, dn, ●●)
    return Ωₖ
end

"""
  A = randh(Val(d), Val(N), Val(Ndn), r::Int,
                    [T=Float64])

  Compute a 'horizontal' core i.e. A[:,1,:] and A[:,2,:] is row-shaped, for k=1.
"""
function randh(::Val{d}, ::Val{Nup}, ::Val{Ndn}, r::Int, ::Type{T}=Float64) where {T<:Number,Nup,Ndn,d}
  ω₁ = randn(T,1,r)
  ω₂ = randn(T,1,r)
  ω₃ = randn(T,1,r)
  ω₄ = randn(T,1,r)
  row_qn = state_qn(Nup,Ndn,d,1)
  col_qn = state_qn(Nup,Ndn,d,2)

  ○○ = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  up = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  dn = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  ●● = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)

  (1,1) in col_qn && (○○[1,1] = ω₁)
  (2,1) in col_qn && (up[1,1] = ω₂)
  (1,2) in col_qn && (dn[1,1] = ω₃)
  (2,2) in col_qn && (●●[1,1] = ω₄)

  Ωₖ = SparseCore{T,Nup,Ndn,d}(1, ○○, up, dn, ●●)
  return Ωₖ
end

"""
  A = randv(Val(d), Val(Nup), Val(Ndn),r::Int,
                    [T=Float64])

  Compute a 'vertical' core i.e. A[:,1,:] and A[:,2,:] is column-shaped, for k=d.
"""
function randv(::Val{d}, ::Val{Nup}, ::Val{Ndn}, r::Int, ::Type{T}=Float64) where {T<:Number,Nup,Ndn,d}
  ω₁ = randn(T,r,1)
  ω₂ = randn(T,r,1)
  ω₃ = randn(T,r,1)
  ω₄ = randn(T,r,1)
  row_qn = state_qn(Nup,Ndn,d,1)
  col_qn = state_qn(Nup,Ndn,d,2)

  ○○ = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  up = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  dn = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)
  ●● = Matrix{Matrix{T}}(undef, Nup+1, Ndn+1)

  for (nup,ndn) in row_qn
    (nup  ,ndn  ) == (Nup,Ndn) && (○○[nup,ndn] = ω₁)
    (nup+1,ndn  ) == (Nup,Ndn) && (up[nup,ndn] = ω₂)
    (nup  ,ndn+1) == (Nup,Ndn) && (dn[nup,ndn] = ω₃)
    (nup+1,ndn+1) == (Nup,Ndn) && (●●[nup,ndn] = ω₄)
  end

  Ωₖ = SparseCore{T,Nup,Ndn,d}(d, ○○, up, dn, ●●)
  return Ωₖ
end


"""
    tt = tt_rand(S, Val(d), Val(Nup), Val(Ndn), r::Vector{Matrix{Int}}, [T=Float64])

Compute a d-dimensional TT-tensor with ranks `r` and entries drawn uniformly from the indexable collection `S` for the cores.
"""
function tt_rand(S, ::Val{d}, ::Val{Nup}, ::Val{Ndn}, r::Vector{Matrix{Int}},
                    ::Type{T}=Float64;
                    orthogonal=:none) where {T<:Number,Nup,Ndn,d}
  @assert eltype(S) <: Number
  @assert T == float(eltype(S))
  @boundscheck @assert (length(r) == d+1) || (length(r) == d-1) && all(axes(r[k]) == (1:Nup+1,1:Ndn+1) for k=1:length(r))

  cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef, d)

  if length(r) == d+1
    for k=1:d
      cores[k] = rand(S, Nup,Ndn,d,k,rank(tt,k),rank(tt,k+1), orthogonal=orthogonal)
    end
  else # length(r) == d-1
    r1 = zeros(Int,Nup+1,Ndn+1)
    r1[1,1] = 1
    cores[1] = rand(S, Nup,Ndn,d,k,r1,r[1], orthogonal=orthogonal)
    for k=2:d-1
      cores[k] = rand(S, Nup,Ndn,d,k,r[k-1],r[k], orthogonal=orthogonal)
    end
    rd = zeros(Int,Nup+1,Ndn+1)
    rd[Nup+1,Ndn+1] = 1
    cores[d] = rand(S, Nup,Ndn,d,k,r[d-1],rd, orthogonal=orthogonal)
  end
  tt = cores2tensor(cores)

  if orthogonal==:right
    tt.orthogonal = true
    tt.corePosition = 1
  elseif orthogonal==:left
    tt.orthogonal = true
    tt.corePosition = d
  end

  check(tt)

  return tt
end

"""
    tt = tt_randn(Val(d), Val(Nup), Val(Ndn), r::Vector{Matrix{Int}}, [T=Float64])

Compute a d-dimensional TT-tensor with ranks `r` and Gaussian distributed entries for the cores.
"""
function tt_randn(::Val{d},::Val{Nup}, ::Val{Ndn}, r::Vector{Matrix{Int}}, ::Type{T}=Float64;
      orthogonal=:none) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert (length(r) in (d-1,d+1)) && all(axes(r[k]) == (1:Nup+1,1:Ndn+1) for k=1:length(r))

  cores = Vector{SparseCore{T,Nup,Ndn,d,Matrix{T}}}(undef, d)

  if length(r) == d+1
    for k=1:d
      cores[k] = randn(Nup,Ndn,d,k,r[k],r[k+1], orthogonal=orthogonal)
    end
  else # length(r) == d-1
    r1 = zeros(Int,Nup+1,Ndn+1)
    r1[1,1] = 1
    cores[1] = randn(Nup,Ndn,d,k,r1,r[1], orthogonal=orthogonal)
    for k=2:d-1
      cores[k] = randn(Nup,Ndn,d,k,r[k-1],r[k], orthogonal=orthogonal)
    end
    rd = zeros(Int,Nup+1,Ndn+1)
    rd[Nup+1,Ndn+1] = 1
    cores[d] = randn(Nup,Ndn,d,k,r[d-1],rd, orthogonal=orthogonal)
  end
  tt = cores2tensor(cores)

  if orthogonal==:right
    tt.orthogonal = true
    tt.corePosition = 1
  elseif orthogonal==:left
    tt.orthogonal = true
    tt.corePosition = d
  end

  check(tt)

  return tt
end

"""
    tt_out = perturbation(tt::TTvector{T,Nup,Ndn,d}, r::Vector{Matrix{Int}}, [ϵ::Float64 = 1e-4])

Compute a d-dimensional TT-tensor perturbation of `tt` with ranks at least `r` and Gaussian distributed entries for the cores.
"""
function perturbation(tt::TTvector{T,Nup,Ndn,d}, r::Vector{Matrix{Int}}, ϵ::Float64 = 1e-4) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert (length(r) in (d-1,d+1)) && all(axes(r[k]) == (1:Nup+1,1:Ndn+1) for k=1:length(r))

  rp = deepcopy(rank(tt))

  if length(r) == d+1
    for k=1:d+1
      for (nup,ndn) in state_qn(Nup,Ndn,d,k)
        rp[k][nup,ndn] = max(1, r[k][nup,ndn] - rank(tt,k,nup,ndn))
      end
    end
  else # length(r) == d-1
    for k=2:d
      for (nup,ndn) in state_qn(Nup,Ndn,d,k)
        rp[k][nup,ndn] = max(1, r[k-1][nup,ndn] - rank(tt,k,nup,ndn))
      end
    end
  end
  p = tt_randn(Val(d),Val(Nup),Val(Ndn),rp)
  lmul!(ϵ*norm(tt)/norm(p), p)
  return tt + p
end

"""
    tt_out = perturbation(tt::TTvector{T,Nup,Ndn,d}, R::Int, [ϵ::Float64 = 1e-4])

Compute a d-dimensional TT-tensor perturbation of `tt` with all block ranks at least `R` and Gaussian distributed entries for the cores.
"""
function perturbation(tt::TTvector{T,Nup,Ndn,d}, R::Int, ϵ::Float64 = 1e-4) where {T<:Number,Nup,Ndn,d}
  r = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for k=2:d, (nup,ndn) in state_qn(Nup,Ndn,d,k)
    r[k][nup,ndn] = R
  end
  return perturbation(tt,r,ϵ)
end