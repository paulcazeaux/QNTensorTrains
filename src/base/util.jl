using OffsetArrays

function chop(sv::Frame{T,N,d,Diagonal{T,Vector{T}}}, ϵ::Float64) where {T<:Number,N,d}
  ranks = similar(sv.row_ranks)

  if ϵ > 0
    svals = diag.(sv.blocks)
    all_svals = sort(reduce(vcat, OffsetArrays.no_offset_view(svals)))
    r = searchsortedlast(cumsum(abs2(s) for s in all_svals), ϵ.^2)    
    leading(s) = (length(s) > 0 ? s[1] : T(0))
    if r < length(all_svals)
      # Chop all singular values strictly smaller than the first one we keep - svals[r+1] 
      # (in case of degeneracy: svals[r] = svals[r+1], they will both be kept)
      cutoff = all_svals[r+1]-eps(T)
      for n in axes(ranks,1)
        ranks[n] = count(>(cutoff), svals[n])
      end
    else # r == length(all_svals), All ranks will be chopped - must keep one
      argmax(leading.(svals))
      for n in axes(ranks,1)
        ranks[n] = (n == n0 ? 1 : 0)
      end
    end
  else # ϵ ≤ 0 
    ranks .= sv.row_ranks
  end
  return ranks
end

function chop(sv::Vector{Frame{T,N,d,Diagonal{T,Vector{T}}}}, ϵ::Float64) where {T<:Number,N,d}
  ranks = Vector{OffsetVector{Int, Vector{Int}}}(undef, length(sv))
  for k=1:length(ranks)
    ranks[k] = similar(sv[k].row_ranks)
  end

  if ϵ > 0
    svals = [diag.(s.blocks) for s in sv]
    all_svals = sort(reduce(vcat, reduce.(vcat, OffsetArrays.no_offset_view.(svals))))
    r = searchsortedlast(cumsum(abs2(s) for s in all_svals), ϵ.^2)
    leading(s) = (length(s) > 0 ? s[1] : T(0))
    if r < length(all_svals)
      # Chop all singular values strictly smaller than the first one we keep - svals[r+1] 
      # (in case of degeneracy: svals[r] = svals[r+1], they will both be kept)
      cutoff = all_svals[r+1]-eps(T)
      for k in axes(ranks,1)
        for n in axes(ranks[k],1)
          ranks[k][n] = count(>(cutoff), svals[k][n])
        end
      # Check that at least one rank is kept for each mode
        if all(ranks[k] .== 0)
          n0 = argmax(leading.(svals[k]))
          ranks[k][n0] = 1
        end
      end
    else # r == length(all_svals), All ranks will be chopped - must keep one
      for k in axes(ranks,1)
        fill!(ranks[k], 0)
        n0 = argmax(leading.(svals[k]))
        ranks[k][n0] = 1
      end
    end
  else # ϵ ≤ 0
    for k in axes(sv,1)
      ranks[k] .= sv[k].row_ranks
    end
  end

  return ranks
end