function chop(sv::Frame{T,Nup,Ndn,d,Diagonal{T,Vector{T}}}, ϵ::Float64) where {T<:Number,Nup,Ndn,d}
  ranks = zeros(Int,Nup+1,Ndn+1)

  if ϵ > 0
    svals = Matrix{Vector{T}}(undef,Nup+1,Ndn+1)
    for (nup,ndn) in qn(sv)
      svals[nup,ndn] = diag(block(sv,nup,ndn))
    end
    all_svals = sort(reduce(vcat, svals[nup,ndn] for (nup,ndn) in qn(sv)))
    r = searchsortedlast(cumsum(abs2(s) for s in all_svals), ϵ.^2)  
    if r < length(all_svals)
      # Chop all singular values strictly smaller than the first one we keep - svals[r+1] 
      # (in case of degeneracy: svals[r] = svals[r+1], they will both be kept)
      cutoff = all_svals[r+1]-eps(T)
      for (nup,ndn) in qn(sv)
        ranks[nup,ndn] = count(>(cutoff), svals[nup,ndn])
      end
    else # r == length(all_svals), All ranks will be chopped - must keep one  
      leading(s) = (length(s) > 0 ? s[1] : T(0))
      n0 = argmax(n->leading(svals[n[1],n[2]]), qn(sv))
      for (nup,ndn) in qn(sv)
        ranks[nup,ndn] = ((nup,ndn) == n0 ? 1 : 0)
      end
    end
  else # ϵ ≤ 0 
    ranks .= sv.row_ranks
  end
  return ranks
end

function chop(sv::Frame{T,Nup,Ndn,d,Diagonal{T,Vector{T}}}, r::Int) where {T<:Number,Nup,Ndn,d}
  svals = Matrix{Vector{T}}(undef,Nup+1,Ndn+1)
  for (nup,ndn) in qn(sv)
    svals[nup,ndn] = diag(block(sv,nup,ndn))
  end
  all_svals      = reduce(vcat, svals[nup,ndn]                              for (nup,ndn) in qn(sv))
  all_svals_tags = reduce(vcat, [(nup,ndn) for _ in length(svals[nup,ndn])] for (nup,ndn) in qn(sv))
  all_svals_tags = all_svals_tags[sortperm(all_svals,rev=true)[1:r]]

  ranks = zeros(Int,Nup+1,Ndn+1)
  for (nup,ndn) in qn(sv)
    ranks[nup,ndn] = count(isequal((nup,ndn)), all_svals_tags)
  end
  return ranks
end

function chop(sv::Vector{Frame{T,Nup,Ndn,d,Diagonal{T,Vector{T}}}}, ϵ::Float64) where {T<:Number,Nup,Ndn,d}
  ranks = [zeros(Int,Nup+1,Ndn+1) for k=1:length(sv)]

  if ϵ > 0
    svals = Vector{Matrix{Vector{T}}}(undef, length(sv))
    for k=1:length(sv)
      svals[k] = Matrix{Vector{T}}(undef,Nup+1,Ndn+1)
      for (nup,ndn) in qn(sv[k])
        svals[k][nup,ndn] = diag(block(sv[k],nup,ndn))
      end
    end
    all_svals = sort(reduce(vcat, reduce(vcat, svals[k][nup,ndn] for (nup,ndn) in qn(sv[k])) for k=1:length(sv)))
    r = searchsortedlast(cumsum(abs2(s) for s in all_svals), ϵ.^2)
    leading(s) = (length(s) > 0 ? s[1] : T(0))
    if r < length(all_svals)
      # Chop all singular values strictly smaller than the first one we keep - svals[r+1] 
      # (in case of degeneracy: svals[r] = svals[r+1], they will both be kept)
      cutoff = all_svals[r+1]-eps(T)
      for k in axes(ranks,1)
        for (nup,ndn) in qn(sv[k])
          ranks[k][nup,ndn] = count(>(cutoff), svals[k][nup,ndn])
        end
      # Check that at least one rank is kept for each mode
        if all(ranks[k] .== 0)
          (nup,ndn) = argmax(qn->leading(svals[k][qn[1],qn[2]]), qn(sv[k]))
          ranks[k][nup,ndn] = 1
        end
      end
    else # r == length(all_svals), All ranks will be chopped - must keep one
      for k in axes(ranks,1)
        fill!(ranks[k], 0)
        (nup,ndn) = argmax(qn->leading(svals[k][qn[1],qn[2]]), qn(sv[k]))
        ranks[k][nup,ndn] = 1
      end
    end
  else # ϵ ≤ 0
    for k in axes(sv,1)
      ranks[k] .= sv[k].row_ranks
    end
  end

  return ranks
end