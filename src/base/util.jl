using OffsetArrays

function chop(sv::OffsetVector{Vector{T},Vector{Vector{T}}}, ϵ::Float64)::OffsetVector{Int64, Vector{Int64}} where T<:Number
  (ϵ ≤ 0) && return length.(sv)

  svals = sort(vcat(sv...))
  r = searchsortedlast(cumsum(svals.^2), ϵ.^2)    

  leading(sv) = (length(sv) > 0 ? sv[1] : T(0))
  if r == length(svals) # All ranks will be chopped - must keep one
    n0 = argmax(leading.(sv))
    return [(n == n0 ? 1 : 0) for n in axes(sv,1)]
  end

  # Chop all singular values strictly smaller than the first one we keep - svals[r+1] 
  # (in case of degeneracy: svals[r] = svals[r+1], they will both be kept)
  cutoff = svals[r+1]-eps(T)
  return [count(>(cutoff), sv[n]) for n in axes(sv,1)]
end

function chop(sv::Vector{OffsetVector{Vector{T},Vector{Vector{T}}}}, ϵ::Float64)::Vector{OffsetVector{Int64, Vector{Int64}}} where T<:Number
  (ϵ ≤ 0) && return [length.(sv[k]) for k in axes(sv,1)]

  svals = sort(vcat([vcat(sv[k]...) for k in axes(sv,1)]...))
  r = searchsortedlast(cumsum(svals.^2), ϵ.^2)

  leading(sv) = (length(sv) > 0 ? sv[1] : T(0))
  if r < length(svals)
    # Chop all singular values strictly smaller than the first one we keep - svals[r+1] 
    # (in case of degeneracy: svals[r] = svals[r+1], they will both be kept)
    cutoff = svals[r+1]-eps(T)
    ranks = [ [count(>(cutoff), sv[k][n]) for n in axes(sv[k],1)] for k in axes(sv,1)]

    # Check that at least one rank is kept for each mode
    for k in axes(sv,1)
      if sum(ranks[k]) == 0
        n0 = argmax(leading.(sv[k]))
        ranks[k] = [(n == n0 ? 1 : 0) for n in axes(sv[k],1)]
      end
    end
  else # All ranks will be chopped - must keep one at each mode, if possible
    nmax = [argmax(leading.(sv[k])) for k in axes(sv,1)]
    ranks = [ [(n == nmax[k] ? 1 : 0) for n in axes(sv[k],1)] for k in axes(sv,1)]
  end

  return ranks
end