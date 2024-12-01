# one body operator with i < j
@test begin
  using QNTensorTrains: Up,Dn,Spin
  T = Float64
  d = 6

  x0 = tt_state((2,), (4,), Val(d)) + tt_state((2,), (5,), Val(d))

  x = AdagᵢAⱼ(x0, (site=1,spin=Up),(site=2,spin=Up))

  ref = -(tt_state((1,),(4,),Val(d))+tt_state((1,),(5,),Val(d)))
  norm(x-ref) ≈ 0
end

##############################

# one body operator with i > j
@test begin
  T = Float64
  d = 6

  x0 = tt_state((6,),(4,), Val(d)) + tt_state((6,),(5,), Val(d)) + tt_state((3,),(2,), Val(d))
  x = AdagᵢAⱼ(x0, (site=5,spin=Dn),(site=4,spin=Dn))
  
  ref = -tt_state((6,),(5,), Val(d))
  norm(x - ref) ≈ 0
end

##############################

# one body operator with i = j
@test begin
  T = Float64
  d = 5
  N = 2

  s1 = (1,4)
  s2 = (1,5)
  s3 = (3,4)

  x0 = tt_state((1,), (4,), Val(d)) + tt_state((1,), (5,), Val(d)) + tt_state((3,), (4,), Val(d))

  x = AdagᵢAⱼ( AdagᵢAⱼ(x0, (site=2,spin=Up),(site=1,spin=Up)), (site=4,spin=Dn),(site=4,spin=Dn))

  ref = -tt_state((2,),(4,), Val(d))
  norm(x - ref, :LR) ≈ 0
end

##############################
### Two body operator test ###
##############################

T = Float64
d = 10
N = rand(3:7)

# Random state with orbitals i,k unoccupied and j,l occupied
function draw_state(i,j,k,l)
  ω = Set((site=i,spin=Up) for i=1:d) ∪ Set((site=i,spin=Dn) for i=1:d)
  pop!(ω, i)
  pop!(ω, k)
  j ∈ ω && pop!(ω, j)
  l ∈ ω && pop!(ω, l)

  return (j, l, [draw!(ω) for ~=3:N]...)
end

function twobody_test(i,j,k,l)
  @assert !(i==k) && !(j==l)
  s = draw_state(i,j,k,l)

  # i,j,k,l =  ((site = 1, spin = Up), (site = 8, spin = Up), (site = 2, spin = Dn), (site = 8, spin = Dn))
  # s = ((site = 8, spin = Up), (site = 8, spin = Dn), (site = 3, spin = Dn), (site = 5, spin = Up), (site = 5, spin = Dn), (site = 6, spin = Dn), (site = 2, spin = Up))

  Nup = count(o->o.spin==Up, s)
  Ndn = count(o->o.spin==Dn, s)
  r = [ zeros(Int,Nup+1,Ndn+1) for k=1:d+1]
  for site=1:d+1, (nup,ndn) in QNTensorTrains.state_qn(Nup,Ndn,d,site)
    r[site][nup,ndn] = (site in (1,d+1) ? 1 : rand(0:5))
  end

  # r = [[1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0], [2 2 0 0 0; 5 2 0 0 0; 0 0 0 0 0; 0 0 0 0 0], [0 2 3 0 0; 5 5 3 0 0; 3 2 2 0 0; 0 0 0 0 0], [5 2 2 2 0; 4 1 0 5 0; 4 0 3 0 0; 2 3 3 2 0], [4 1 0 3 3; 1 0 1 5 3; 4 1 2 1 3; 3 1 0 2 4], [0 2 1 4 2; 0 5 4 5 1; 3 1 5 1 2; 3 5 5 1 1], [4 5 1 4 0; 2 1 4 2 5; 4 2 4 3 2; 2 1 0 1 3], [0 1 2 0 1; 0 3 3 2 1; 0 3 5 1 2; 0 1 3 0 0], [0 0 0 0 0; 0 0 2 1 3; 0 0 2 2 3; 0 0 0 5 0], [0 0 0 0 0; 0 0 0 0 0; 0 0 0 2 5; 0 0 0 0 4], [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 1]]

  ref_s = (i, k, s[3:end]...)
  p = round(tt_randn(Val(d),Val(Nup),Val(Ndn),r)); rmul!(p, 1/norm(p))
  if j.spin == l.spin
    Sjl = delete!(delete!(Set((site=i,spin=j.spin) for i=1:d),j),l)
    x0 = tt_state(tuple(collect(i.site for i in s if i.spin==Up)...), tuple(collect(i.site for i in s if i.spin==Dn)...), Val(d)) + AdagᵢAⱼ(p, rand(Sjl), j) + AdagᵢAⱼ(p, rand(Sjl), l) # Random perturbation that should go to zero
  else
    Sj = delete!(Set((site=i,spin=j.spin) for i=1:d),j)
    Sl = delete!(Set((site=i,spin=l.spin) for i=1:d),l)
    x0 = tt_state(tuple(collect(i.site for i in s if i.spin==Up)...), tuple(collect(i.site for i in s if i.spin==Dn)...), Val(d)) + AdagᵢAⱼ(p, rand(Sj), j) + AdagᵢAⱼ(p, rand(Sl), l) # Random perturbation that should go to zero
  end
  x = AdagᵢAdagₖAₗAⱼ(x0, i,j,k,l)

  QNTensorTrains.check(x)

  if typeof(dot(x,x))!=Float64 || dot(x,x) == 0.0
    @show Nup, Ndn, dot(x,x)
    @show i,j,k,l
    @show s
    @show r
    @show rank(x)
  end
  # Count electrons on each site
  nx = [ (dot(x, AdagᵢAⱼ(x, (site=k,spin=Up), (site=k,spin=Up)))+dot(x, AdagᵢAⱼ(x, (site=k,spin=Dn), (site=k,spin=Dn)))) / dot(x,x) for k=1:d]
  nref = [((site=k,spin=Up) ∈ ref_s && (site=k,spin=Dn) ∈ ref_s ? 2 : 
           (site=k,spin=Up) ∈ ref_s || (site=k,spin=Dn) ∈ ref_s ? 1 : 0) for k=1:d]
 
  test = isapprox(norm(nx.-nref),  0;  atol=10eps()*norm(x0))

  if !(test)
    @show i,j,k,l
    @show nx, nref
    @show norm(nx.-nref)/norm(x0)
  end

  return test
end

ω = Set((site=i,spin=Up) for i=1:d) ∪ Set((site=i,spin=Dn) for i=1:d)
S = sort([draw!(ω) for ~ = 1:4])

for i in S, j in filter(j->j.spin==i.spin,S), k in filter(≠(i), S), l in filter(l->(l≠j && l.spin==k.spin), S)
  @test twobody_test(i,j,k,l)
end

