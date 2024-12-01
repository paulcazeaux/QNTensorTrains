"""
    conj!(core::SparseCore{T,Nup,Ndn,d})

Transform a TT-tensor `tt` to its complex conjugate in-place.
"""
function Base.conj!(core::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  for (lup,ldn) in row_qn(core)
    (lup  ,ldn  ) in col_qn(core) && conj!(○○(core,lup,ldn))
    (lup+1,ldn  ) in col_qn(core) && conj!(up(core,lup,ldn))
    (lup  ,ldn+1) in col_qn(core) && conj!(dn(core,lup,ldn))
    (lup+1,ldn+1) in col_qn(core) && conj!(●●(core,lup,ldn))
  end
  return core
end

"""
    real(core::SparseCore{T,Nup,Ndn,d})

Compute the real part of `core`.
"""
function Base.real(v::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  w = similar(v)
  for (lup,ldn) in row_qn(v)
    (lup  ,ldn  ) in col_qn(v) && ○○(w,lup,ldn) .= real(○○(v,lup,ldn))
    (lup+1,ldn  ) in col_qn(v) && up(w,lup,ldn) .= real(up(v,lup,ldn))
    (lup  ,ldn+1) in col_qn(v) && dn(w,lup,ldn) .= real(dn(v,lup,ldn))
    (lup+1,ldn+1) in col_qn(v) && ●●(w,lup,ldn) .= real(●●(v,lup,ldn))
  end
  return w
end

"""
    imag(core::SparseCore{T,Nup,Ndn,d})

Compute the imag part of `core`.
"""
function Base.imag(v::SparseCore{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  w = similar(v)
  for (lup,ldn) in row_qn(v)
    (lup  ,ldn  ) in col_qn(v) && ○○(w,lup,ldn) .= imag(○○(v,lup,ldn))
    (lup+1,ldn  ) in col_qn(v) && up(w,lup,ldn) .= imag(up(v,lup,ldn))
    (lup  ,ldn+1) in col_qn(v) && dn(w,lup,ldn) .= imag(dn(v,lup,ldn))
    (lup+1,ldn+1) in col_qn(v) && ●●(w,lup,ldn) .= imag(●●(v,lup,ldn))
  end
  return w
end
"""
    conj(tt::TTvector{T,Nup,Ndn,d})

Compute the complex conjugate of TT-tensor `tt`.
"""
function Base.conj(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  tt1 = deepcopy(tt)
  conj!(tt1)
  return tt1
end

"""
    conj!(tt::TTvector{T,Nup,Ndn,d})

Transform a TT-tensor `tt` to its complex conjugate in-place.
"""
function Base.conj!(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  for j=1:d
    conj!(tt.cores[i])
  end
  return tt
end

"""
    real(tt::TTvector{T,Nup,Ndn,d})

Compute the real part of `tt`.
"""
function Base.real(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if d == 1
    cores = [real(core(tt,1))]
  else
    cores = [SparseCore{T,Nup,Ndn,d}(k, (k==1 ? 1 : 2) .* rank(tt,k), (k==d ? 1 : 2) .* rank(tt,k+1)) for k=1:d]

    # k==1
    for n in it_○○(cores[1])
      ○○(cores[1], n) .= hcat( real(○○(cores(tt,1))), imag(○○(cores(tt,1))) )
    end
    for n in it_up(cores[1])
      up(cores[1], n) .= hcat( real(up(cores(tt,1))), imag(up(cores(tt,1))) )
    end
    for n in it_dn(cores[1])
      dn(cores[1], n) .= hcat( real(dn(cores(tt,1))), imag(dn(cores(tt,1))) )
    end
    for n in it_●●(cores[1])
      ●●(cores[1], n) .= hcat( real(●●(cores(tt,1))), imag(●●(cores(tt,1))) )
    end

    for k = 2:d-1
      for n in it_○○(cores[k])
        ○○(cores[k], n) .= hvcat(2,  real(○○(cores(tt,k))), imag(○○(cores(tt,k))),
                                    -imag(○○(cores(tt,k))), real(○○(cores(tt,k))) )
      end
      for n in it_up(cores[k])
        up(cores[k], n) .= hvcat(2,  real(up(cores(tt,k))), imag(up(cores(tt,k))),
                                    -imag(up(cores(tt,k))), real(up(cores(tt,k))) )
      end
      for n in it_dn(cores[k])
        dn(cores[k], n) .= hvcat(2,  real(dn(cores(tt,k))), imag(dn(cores(tt,k))),
                                    -imag(dn(cores(tt,k))), real(dn(cores(tt,k))) )
      end
      for n in it_●●(cores[k])
        ●●(cores[k], n) .= hvcat(2,  real(●●(cores(tt,k))), imag(●●(cores(tt,k))),
                                    -imag(●●(cores(tt,k))), real(●●(cores(tt,k))) )
      end
    end
    # k==d
    for n in it_○○(cores[d])
      ○○(cores[d], n) .= vcat( real(○○(cores(tt,d))), -imag(○○(cores(tt,d))) )
    end
    for n in it_up(cores[d])
      up(cores[d], n) .= vcat( real(up(cores(tt,d))), -imag(up(cores(tt,d))) )
    end
    for n in it_dn(cores[d])
      dn(cores[d], n) .= vcat( real(dn(cores(tt,d))), -imag(dn(cores(tt,d))) )
    end
    for n in it_●●(cores[d])
      ●●(cores[d], n) .= vcat( real(●●(cores(tt,d))), -imag(●●(cores(tt,d))) )
    end
  end
  tt_r = cores2tensor(cores)
  return tt_r
end

"""
    imag(tt::TTvector{T,Nup,Ndn,d})

Compute the imaginary part of `tt`.
"""
function Base.imag(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if d == 1
    cores = [imag(core(tt,1))]
  else
    cores = [SparseCore{T,Nup,Ndn,d}(k, (k==1 ? 1 : 2) .* rank(tt,k), (k==d ? 1 : 2) .* rank(tt,k+1)) for k=1:d]

    # k==1
    for n in it_○○(cores[1])
      ○○(cores[1], n) .= hcat( real(○○(cores(tt,1))), imag(○○(cores(tt,1))) )
    end
    for n in it_up(cores[1])
      up(cores[1], n) .= hcat( real(up(cores(tt,1))), imag(up(cores(tt,1))) )
    end
    for n in it_dn(cores[1])
      dn(cores[1], n) .= hcat( real(dn(cores(tt,1))), imag(dn(cores(tt,1))) )
    end
    for n in it_●●(cores[1])
      ●●(cores[1], n) .= hcat( real(●●(cores(tt,1))), imag(●●(cores(tt,1))) )
    end

    for k = 2:d-1
      for n in it_○○(cores[k])
        ○○(cores[k], n) .= hvcat(2,  real(○○(cores(tt,k))), imag(○○(cores(tt,k))),
                                    -imag(○○(cores(tt,k))), real(○○(cores(tt,k))) )
      end
      for n in it_up(cores[k])
        up(cores[k], n) .= hvcat(2,  real(up(cores(tt,k))), imag(up(cores(tt,k))),
                                    -imag(up(cores(tt,k))), real(up(cores(tt,k))) )
      end
      for n in it_dn(cores[k])
        dn(cores[k], n) .= hvcat(2,  real(dn(cores(tt,k))), imag(dn(cores(tt,k))),
                                    -imag(dn(cores(tt,k))), real(dn(cores(tt,k))) )
      end
      for n in it_●●(cores[k])
        ●●(cores[k], n) .= hvcat(2,  real(●●(cores(tt,k))), imag(●●(cores(tt,k))),
                                    -imag(●●(cores(tt,k))), real(●●(cores(tt,k))) )
      end
    end
    # k==d
    for n in it_○○(cores[d])
      ○○(cores[d], n) .= vcat( imag(○○(cores(tt,d))), real(○○(cores(tt,d))) )
    end
    for n in it_up(cores[d])
      up(cores[d], n) .= vcat( imag(up(cores(tt,d))), real(up(cores(tt,d))) )
    end
    for n in it_dn(cores[d])
      dn(cores[d], n) .= vcat( imag(dn(cores(tt,d))), real(dn(cores(tt,d))) )
    end
    for n in it_●●(cores[d])
      ●●(cores[d], n) .= vcat( imag(●●(cores(tt,d))), real(●●(cores(tt,d))) )
    end
  end
  tt_i = cores2tensor(cores)
  return tt_i
end


"""
    realimag(tt::TTvector{T,Nup,Ndn,d})

Compute a tuple containing the real and imaginary part of `tt`.
"""
function realimag(tt::TTvector{T,Nup,Ndn,d}) where {T<:Number,Nup,Ndn,d}
  if d == 1
    tt_r = cores2tensor([real(core(tt,1))])
    tt_i = cores2tensor([imag(core(tt,1))])
  else
    cores = [SparseCore{T,Nup,Ndn,d}(k, (k==1 ? 1 : 2) .* rank(tt,k), (k==d ? 1 : 2) .* rank(tt,k+1)) for k=1:d]

    # k==1
    for n in it_○○(cores[1])
      ○○(cores[1], n) .= hcat( real(○○(cores(tt,1))), imag(○○(cores(tt,1))) )
    end
    for n in it_up(cores[1])
      up(cores[1], n) .= hcat( real(up(cores(tt,1))), imag(up(cores(tt,1))) )
    end
    for n in it_dn(cores[1])
      dn(cores[1], n) .= hcat( real(dn(cores(tt,1))), imag(dn(cores(tt,1))) )
    end
    for n in it_●●(cores[1])
      ●●(cores[1], n) .= hcat( real(●●(cores(tt,1))), imag(●●(cores(tt,1))) )
    end

    for k = 2:d-1
      for n in it_○○(cores[k])
        ○○(cores[k], n) .= hvcat(2,  real(○○(cores(tt,k))), imag(○○(cores(tt,k))),
                                    -imag(○○(cores(tt,k))), real(○○(cores(tt,k))) )
      end
      for n in it_up(cores[k])
        up(cores[k], n) .= hvcat(2,  real(up(cores(tt,k))), imag(up(cores(tt,k))),
                                    -imag(up(cores(tt,k))), real(up(cores(tt,k))) )
      end
      for n in it_dn(cores[k])
        dn(cores[k], n) .= hvcat(2,  real(dn(cores(tt,k))), imag(dn(cores(tt,k))),
                                    -imag(dn(cores(tt,k))), real(dn(cores(tt,k))) )
      end
      for n in it_●●(cores[k])
        ●●(cores[k], n) .= hvcat(2,  real(●●(cores(tt,k))), imag(●●(cores(tt,k))),
                                    -imag(●●(cores(tt,k))), real(●●(cores(tt,k))) )
      end
    end
    # k==d - real part
    for n in it_○○(cores[d])
      ○○(cores[d], n) .= vcat( real(○○(cores(tt,d))), -imag(○○(cores(tt,d))) )
    end
    for n in it_up(cores[d])
      up(cores[d], n) .= vcat( real(up(cores(tt,d))), -imag(up(cores(tt,d))) )
    end
    for n in it_dn(cores[d])
      dn(cores[d], n) .= vcat( real(dn(cores(tt,d))), -imag(dn(cores(tt,d))) )
    end
    for n in it_●●(cores[d])
      ●●(cores[d], n) .= vcat( real(●●(cores(tt,d))), -imag(●●(cores(tt,d))) )
    end
    tt_r = cores2tensor(cores)

    cores[d] = SparseCore{T,Nup,Ndn,d}(k, 2 .*rank(tt,d), rank(tt,d+1))
    # k==d - imaginary part
    for n in it_○○(cores[d])
      ○○(cores[d], n) .= vcat( imag(○○(cores(tt,d))), real(○○(cores(tt,d))) )
    end
    for n in it_up(cores[d])
      up(cores[d], n) .= vcat( imag(up(cores(tt,d))), real(up(cores(tt,d))) )
    end
    for n in it_dn(cores[d])
      dn(cores[d], n) .= vcat( imag(dn(cores(tt,d))), real(dn(cores(tt,d))) )
    end
    for n in it_●●(cores[d])
      ●●(cores[d], n) .= vcat( imag(●●(cores(tt,d))), real(●●(cores(tt,d))) )
    end
    tt_i = cores2tensor(cores)
  end

  return (tt_r, tt_i)
end