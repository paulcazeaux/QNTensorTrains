@test begin
  T = Float64
  d = 20
  N = 10

  s1 = [2,4,6,7,8,10,13,14,15,18]
  s2 = [1,3,5,7,9,11,13,15,17,19]

  for k=1:d
    C = SparseCore{T,N,d}(k∈s2,sum(k.>s2),k)  ⊕ 
        SparseCore{T,N,d}(k∈s2,sum(k.>s2),k)
  end
  true
end
