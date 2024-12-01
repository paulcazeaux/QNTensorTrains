@test begin
  T = Float64
  d = 20
  N = 20
  Sz = 0//2
  Nup = Int(N+2Sz)÷2
  Ndn = Int(N-2Sz)÷2

  s1 = [2,4,6,7,8,10,13,14,15,18]
  s2 = [1,3,5,7,9,11,13,15,17,19]

  for k=1:d
    C = SparseCore{T,Nup,Ndn,d}(k∈s1,sum(k.>s1),k∈s2,sum(k.>s2),k)  ⊕ 
        SparseCore{T,Nup,Ndn,d}(k∈s2,sum(k.>s2),k∈s1,sum(k.>s1),k)
  end
  true
end
