"""
    conj!(core::SparseCore{T,N,d})

Transform a TT-tensor `tt` to its complex conjugate in-place.
"""
function Base.conj!(core::SparseCore{T,N,d}) where {T<:Number,N,d}
  for l in axes(core,1), r in (l:l+1)∩axes(core,3)
    if isnonzero(core[l,r])
      core[l,r].factor = conj(factor(core[l,r]))
      conj!(data(core[l,r]))
    end
  end
  return core
end

"""
    real(core::SparseCore{T,N,d})

Compute the real part of `core`.
"""
function Base.real(v::SparseCore{T,N,d}) where {T<:Number,N,d}
  w = similar(v)
  for l in axes(v,1), r in (l:l+1)∩axes(v,3)
    if isnonzero(v[l,r])
      w[l,r] = real(factor(v[l,r])), real(data(v[l,r]))
    end
  end
  return w
end

"""
    imag(core::SparseCore{T,N,d})

Compute the imag part of `core`.
"""
function Base.imag(v::SparseCore{T,N,d}) where {T<:Number,N,d}
  w = similar(v)
  for l in axes(v,1), r in (l:l+1)∩axes(v,3)
    if isnonzero(v[l,r])
      w[l,r] = imag(factor(v[l,r])), imag(data(v[l,r]))
    end
  end
  return w
end
"""
    conj(tt::TTvector{T,N,d})

Compute the complex conjugate of TT-tensor `tt`.
"""
function Base.conj(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  tt1 = deepcopy(tt)
  conj!(tt1)
  return tt1
end

"""
    conj!(tt::TTvector{T,N,d})

Transform a TT-tensor `tt` to its complex conjugate in-place.
"""
function Base.conj!(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  for j=1:d
    conj!(tt.cores[i])
  end
  return tt
end

"""
    real(tt::TTvector{T,N,d})

Compute the real part of `tt`.
"""
function Base.real(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  if d == 1
    cores = [real(core(tt,1))]
  else
    cores = [SparseCore{T,N,d}(k) for k=1:d]

    # k==1
    cores[1].row_ranks = rank(tt,1)
    cores[1].col_ranks = 2 .* rank(tt,2)
    cores[1].unoccupied = hcat.(real.(core(tt,1).unoccupied), imag.(core(tt,1).unoccupied))
    cores[1].occupied   = hcat.(real.(core(tt,1).occupied  ), imag.(core(tt,1).occupied  ))

    for k = 2:d-1
      cores[k].row_ranks = 2 .* rank(tt,k)
      cores[k].col_ranks = 2 .* rank(tt,k+1)
      cores[k].unoccupied =  hvcat.(2, real.(core(tt,k).unoccupied), imag.(core(tt,k).unoccupied),
                                      -imag.(core(tt,k).unoccupied), real.(core(tt,k).unoccupied) )
      cores[k].occupied   =  hvcat.(2, real.(core(tt,k).occupied  ), imag.(core(tt,k).occupied  ),
                                      -imag.(core(tt,k).occupied  ), real.(core(tt,k).occupied  ) )
    end
    # k==d
    cores[d].row_ranks = 2 .* rank(tt,d)
    cores[d].col_ranks = rank(tt,d+1)
    cores[d].unoccupied = vcat.(real.(core(tt,d).unoccupied), -imag.(core(tt,d).unoccupied))
    cores[d].occupied   = vcat.(real.(core(tt,d).occupied  ), -imag.(core(tt,d).occupied  ))
  end
  tt_r = cores2tensor(cores)
  return tt_r
end

"""
    imag(tt::TTvector{T,N,d})

Compute the imaginary part of `tt`.
"""
function Base.imag(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  if d == 1
    cores = [imag(core(tt,1))]
  else
    cores = [SparseCore{T,N,d}(k) for k=1:d]

    # k==1
    cores[1].row_ranks = rank(tt,1)
    cores[1].col_ranks = 2 .* rank(tt,2)
    cores[1].unoccupied = hcat.(real.(core(tt,1).unoccupied), imag.(core(tt,1).unoccupied))
    cores[1].occupied   = hcat.(real.(core(tt,1).occupied  ), imag.(core(tt,1).occupied  ))

    for k = 2:d-1
      cores[k].row_ranks = 2 .* rank(tt,k)
      cores[k].col_ranks = 2 .* rank(tt,k+1)
      cores[k].unoccupied =  hvcat.(2, real.(core(tt,k).unoccupied), imag.(core(tt,k).unoccupied),
                                      -imag.(core(tt,k).unoccupied), real.(core(tt,k).unoccupied) )
      cores[k].occupied   =  hvcat.(2, real.(core(tt,k).occupied  ), imag.(core(tt,k).occupied  ),
                                      -imag.(core(tt,k).occupied  ), real.(core(tt,k).occupied  ) )
    end
    # k==d
    cores[d].row_ranks = 2 .* rank(tt,d)
    cores[d].col_ranks = rank(tt,d+1)
    cores[d].unoccupied = vcat.(imag.(core(tt,d).unoccupied), real.(core(tt,d).unoccupied))
    cores[d].occupied   = vcat.(imag.(core(tt,d).occupied  ), real.(core(tt,d).occupied  ))
  end
  tt_i = cores2tensor(cores)
  return tt_i
end


"""
    realimag(tt::TTvector{T,N,d})

Compute a tuple containing the real and imaginary part of `tt`.
"""
function realimag(tt::TTvector{T,N,d}) where {T<:Number,N,d}
  if d == 1
    tt_r = cores2tensor([real(core(tt,1))])
    tt_i = cores2tensor([imag(core(tt,1))])
    return (tt_r, tt_i)
  else
    cores = [SparseCore{T,N,d}(k) for k=1:d]

    # k==1
    cores[1].row_ranks = rank(tt,1)
    cores[1].col_ranks = 2 .* rank(tt,2)
    cores[1].unoccupied = hcat.(real.(core(tt,1).unoccupied), imag.(core(tt,1).unoccupied))
    cores[1].occupied   = hcat.(real.(core(tt,1).occupied  ), imag.(core(tt,1).occupied  ))

    for k = 2:d-1
      cores[k].row_ranks = 2 .* rank(tt,k)
      cores[k].col_ranks = 2 .* rank(tt,k+1)
      cores[k].unoccupied =  hvcat.(2, real.(core(tt,k).unoccupied), imag.(core(tt,k).unoccupied),
                                      -imag.(core(tt,k).unoccupied), real.(core(tt,k).unoccupied) )
      cores[k].occupied   =  hvcat.(2, real.(core(tt,k).occupied  ), imag.(core(tt,k).occupied  ),
                                      -imag.(core(tt,k).occupied  ), real.(core(tt,k).occupied  ) )
    end
    # k==d
    cores[d].row_ranks = 2 .* rank(tt,d)
    cores[d].col_ranks = rank(tt,d+1)
    
    cores[d].unoccupied = vcat.(real.(core(tt,d).unoccupied), -imag.(core(tt,d).unoccupied))
    cores[d].occupied   = vcat.(real.(core(tt,d).occupied  ), -imag.(core(tt,d).occupied  ))
    tt_r = cores2tensor(deepcopy(cores))

    cores[d].unoccupied = vcat.(imag.(core(tt,d).unoccupied), real.(core(tt,d).unoccupied))
    cores[d].occupied   = vcat.(imag.(core(tt,d).occupied  ), real.(core(tt,d).occupied  ))
    tt_i = cores2tensor(cores)
  end
  return (tt_r, tt_i)
end