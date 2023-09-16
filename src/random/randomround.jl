"""
    roundRandOrth!(tt::TTvector{T,N,d}, target_r::Array{Int,1})

Truncates the ranks of `tt` with specified ranks target_rank.
"""
function roundRandOrth!(tt::TTvector{T,N,d}, target_r::Vector{OffsetVector{Int,Vector{Int}}}) where {T<:Number,N,d}
  # Approximate sTT-tensor with another one with specified accuracy
  # round!(tt, eps)       Approximate TT-tensor with relative accuracy eps
  # round!(tt, eps, rmax) Approximate TT-tensor with relative accuracy and maximal rank rmax,
  #                       which may be an array of ranks or a number.

  @assert target_r[1][0] == rank(tt,1,0)
  @assert target_r[d+1][N] == rank(tt,d+1,N)
  r = rank(tt)
  Ω = tt_rand(d,N,target_r, T)

  # Precompute partial projections W
  W = [[zeros(T,0,0) for n in axis(core(tt,k),3)] for k=1:d]
  W[d][N] = [1;;]

  for k=d-1:-1:1
    for l in axis(core(tt,k+1),1)
      W[k][l] = sum(
                  core(tt,k+1)[l,r] * W[k+1][r] * adjoint(core(tt0,k+1)[l,r]) 
                  for r in axis(core(tt1,k+1),3)∩(l:l+1)
                )
    end
  end

  for k=1:d-1
    C = [zeros(T,0,0) for n in axis(A,3)]
    rank = [T(0) for n in axis(A,3)]

    for n in axis(A,3)
      Q = my_qc!(core(tt,k)[n,:vertical] * W[k])
      C[n] = Q' * core(tt,i,:vertical)
      rank[n] = size(Q,2)
      core(tt,k)[n,:vertical] = Q
    end
    mult!(C, core(tt,k+1))
    tt.r[k+1] .= rank
  end

  tt.orthogonal = true
  tt.corePosition = d
  tt.over = 0

  return tt
end

function roundRandOrth(tt::TTvector, target_r::Vector{OffsetVector{Int,Vector{Int}}})
  return roundRandOrth!(deepcopy(tt), target_r)
end