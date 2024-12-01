"""
    dot(tt1::TTvector{T,Nup,Ndn,d}, tt2::TTvector{T,Nup,Ndn,d}; [orthogonalize=false])

Compute the scalar product of two TT-tensors.
In general, returns a 4D tensor of sizes `(r1[1],r2[1],r1[d+1],r2[d+1])`.
  * if `r1[1]=r1[1]=1`, returns a matrix of size `(r1[d+1],r2[d+1])`.
  * if `r1[d+1]=r1[d+1]=1`, returns a matrix of size `(r1[1],r2[1])`.
  * if `r1[1]=r1[1]=r1[d+1]=r1[d+1]=1`, returns a scalar.

If `orthogonalize=true` is specified, performs the left-to-right QRs of `tt1`, `tt2` before the scalar product.
It increases the accuracy in some cases.
"""
function LinearAlgebra.dot(tt1::TTvector{T,Nup,Ndn,d}, tt2::TTvector{T,Nup,Ndn,d}; orthogonalize::Bool=false) where {T<:Number,Nup,Ndn,d}
  if (orthogonalize)
    rightOrthogonalize!(tt1)
    rightOrthogonalize!(tt2)
  end

  if rank(tt1,1,1)==rank(tt2,1,1)==1
    p = IdFrame(Val(d), Val(Nup), Val(Ndn), 1)
    for k=1:d
      p = adjoint(core(tt1,k)) * (p * core(tt2,k))
    end
    if rank(tt1,d+1,1)==rank(tt2,d+1,1)==1
      return p[1][1,1]
    else
      return p[1]
    end
  else
    p = [ (nup==ndn==1 ? reshape( I(rank(tt1,1,1)*rank(tt2,1,1)), 
                         (rank(tt1,1,1), rank(tt2,1,1), rank(tt1,1,1), rank(tt2,1,1)) )
                       : zeros(T,0,0) ) for nup in 1:Nup+1, ndn in 1:Ndn+1]
    for k=1:d
      C₁ₖ = core(tt1,k)
      C₂ₖ = core(tt2,k)
      for nup in reverse(1:Nup+1), ndn in reverse(1:Ndn+1)
        if (nup,ndn) in col_qn(C₁ₖ)
          if (nup,ndn) in row_qn(C₁ₖ)
            p[nup,ndn] = reshape( adjoint(○○(C₁ₖ,nup,ndn)) * reshape(p[nup,ndn], rank(tt1,k,(nup,ndn)), :), 
                                          :, rank(tt2,k,(nup,ndn)) ) * ○○(C₂ₖ,nup,ndn)
          else
            p[nup,ndn] = zeros(T, rank(tt1,k+1,(nup,ndn)), rank(tt2,1,1), rank(tt1,1,1), rank(tt2,k+1,(nup,ndn)))
          end
          (nup-1,ndn  ) in row_qn(C₁ₖ) && p[nup,ndn] .+= reshape( adjoint(up(C₁ₖ,nup-1,ndn  )) * reshape(p[nup-1,ndn  ], rank(tt1,k,(nup-1,ndn  )), :), 
                                          :, rank(tt2,k,(nup-1,ndn  )) ) * up(C₂ₖ,nup-1,ndn  )
          (nup  ,ndn-1) in row_qn(C₁ₖ) && p[nup,ndn] .+= reshape( adjoint(dn(C₁ₖ,nup  ,ndn-1)) * reshape(p[nup  ,ndn-1], rank(tt1,k,(nup  ,ndn-1)), :), 
                                          :, rank(tt2,k,(nup  ,ndn-1)) ) * dn(C₂ₖ,nup  ,ndn-1)
          (nup-1,ndn-1) in row_qn(C₁ₖ) && p[nup,ndn] .+= reshape( adjoint(●●(C₁ₖ,nup-1,ndn-1)) * reshape(p[nup-1,ndn-1], rank(tt1,k,(nup-1,ndn-1)), :), 
                                          :, rank(tt2,k,(nup-1,ndn-1)) ) * ●●(C₂ₖ,nup-1,ndn-1)
        end
      end
    end

    P = permutedims(p[Nup+1,Ndn+1], [1,4,3,2])

    if rank(tt1,d+1,N)==rank(tt2,d+1,N)==1
      return dropdims(P, dims=(3,4))
    else
      return P
    end
  end
end

function step_core_left_norm!!(tt::TTvector{T,Nup,Ndn,d}, k::Int, keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert 2 ≤ k ≤ d

  if (keepRank)
    L = lq!(core(tt,k))
    nrm = norm(norm.(L))
    if nrm != zero(T)
      for n in axes(core(tt,k),1)
        L[k] ./= nrm
      end
    end
    rmul!(core(tt,k-1), L)
  else
    Qk, C, ranks = cq!(core(tt,k))
    nrm = norm(C)
    if nrm != zero(T)
      for n in axes(core(tt,k),1)
        C[n] ./= nrm
      end
    end
    set_cores!(tt, core(tt,k-1) * C, Qk)
  end
  
  if tt.orthogonal && (tt.corePosition == k)
    tt.corePosition = k-1
  end

  return nrm
end

function step_core_right_norm!!(tt::TTvector{T,Nup,Ndn,d}, k::Int, keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  @boundscheck @assert 1 ≤ k ≤ d-1

  if (keepRank)
    R = qr!(core(tt,k))
    nrm = norm(norm.(R))
    if nrm != zero(T)
      for n in axes(core(tt,k),3)
        R[n] ./= nrm
      end
    end
    lmul!(R, core(tt,k+1))
  else
    Qk, C, ranks = qc!(core(tt,k))
    nrm = norm(C)
    if nrm != zero(T)
      for n in axes(core(tt,k),3)
        C[n] ./= nrm
      end
    end
    set_cores!(tt, Qk, C * core(tt,k+1))
  end

  if tt.orthogonal && (tt.corePosition == k)
    tt.corePosition = k+1
  end

  return nrm
end

function rightOrthonormalize!!(tt::TTvector{T,Nup,Ndn,d}, keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  nrm=[T(1) for k=1:d]

  for k=(tt.orthogonal ? tt.corePosition : d):-1:2
    nrm[k] = step_core_left_norm!!(tt,k,keepRank)
  end
  nrm[1] = norm(core(tt,1))
  rmul!(core(tt,1), inv(nrm[1]))
  tt.orthogonal = true
  tt.corePosition = 1
  return tt, nrm
end

function leftOrthonormalize!!(tt::TTvector{T,Nup,Ndn,d}, keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  nrm=[T(1) for k=1:d]

  for k=(tt.orthogonal ? tt.corePosition : 1):d-1
    nrm[k] = step_core_right_norm!!(tt,k,keepRank)
  end
  nrm[d] = norm(core(tt,d))
  rmul!(core(tt,d), inv(nrm[d]))
  tt.orthogonal = true
  tt.corePosition = d
  return tt, nrm
end

"""
    norm(tt::TTvector{T,Nup,Ndn,d}; orthogonalize::Symbol=:none)

Compute the Frobenius norm of the TT-tensor.

Optionally you can specify the keywork `orthogonalize` to use a right-left sweep of QRs (`:RL`) or a left-right sweep (`:LR`) to obtain a more accurate result.
"""
function LinearAlgebra.norm(tt::TTvector{T,Nup,Ndn,d}, orthogonalize::Symbol=:none, keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  if tt.orthogonal
    return norm(core(tt, tt.corePosition))
  elseif orthogonalize==:none  #Compute dot product of tt with itself
    return sqrt(abs(sum(dot(tt, tt))))
  elseif orthogonalize==:LR
    tt, nrm = leftOrthonormalize!!(tt, keepRank)
  elseif orthogonalize==:RL
    tt, nrm = rightOrthonormalize!!(tt, keepRank)
  else
    error("Wrong argument for `orthogonalize` keyword in norm")
  end

  for i=1:d
    tt.cores[i] *= exp(sum(log.(nrm[k]) for k=1:d)/d)
    tt.orthogonal = false
  end
  return exp(sum(log.(nrm[k]) for k=1:d))
end

"""
    lognorm(tt::TTvector{T,Nup,Ndn,d}; orthogonalize::Symbol=:none)

Compute log10 of the Frobenius norm of the TT-tensor.

Optionally you can specify the keywork `orthogonalize` to use a right-left sweep of QRs (`:RL`) or a left-right sweep (`:LR`) to obtain a more accurate result.
"""
function lognorm(tt::TTvector{T,Nup,Ndn,d}, orthogonalize::Symbol=:none, keepRank::Bool=false) where {T<:Number,Nup,Ndn,d}
  if tt.orthogonal
    return log10(norm(core(tt, tt.corePosition)))
  elseif orthogonalize==:none
    return 0.5 * log10(sum(dot(tt, tt)))
  elseif orthogonalize==:LR
    tt, nrm = leftOrthonormalize!!(tt, keepRank)
  elseif orthogonalize==:RL
    tt, nrm = rightOrthonormalize!!(tt, keepRank)
  else
    error("Wrong argument for `orthogonalize` keyword in norm")
  end

  nrm[d] = norm(core(tt, d))
  if (nrm[d]!= 0)
    rmul!(tt.cores[d], inv(nrm[d]))
  end
  for i=1:d
    tt.cores[i] .*= exp(sum(log.(nrm))/d)
    tt.orthogonal = false
  end
  return sum(log10.(nrm))
end