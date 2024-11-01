# one body operator with i < j
@test begin
  T = Float64
  d = 6
  N = 2

  s1 = (2,4)
  s2 = (2,5)

  x0 = tt_state(s1, Val(d)) + tt_state(s2, Val(d))

  x = AdagᵢAⱼ(x0, 1,2)

  ref = (tt_state((1,4),Val(d))+tt_state((1,5),Val(d)))
  norm(x-ref) ≈ 0
end


# one body operator with i > j
@test begin
  T = Float64
  d = 6
  N = 2

  s1 = (6,4)
  s2 = (6,5)
  s3 = (2,3)

  x0 = tt_state(s1, Val(d)) + tt_state(s2, Val(d)) + tt_state(s3, Val(d))
  x = AdagᵢAⱼ(x0, 5,4)
  
  ref = tt_state((5,6), Val(d))
  norm(x - ref) ≈ 0
end

##############################
### Two body operator test ###
##############################

T = Float64
d = 10
N = rand(3:7)

# Random state with sites i,j unoccupied and k,l occupied
function draw_state(i,j,k,l)
  @assert i<j && k<l
  ω = Set(1:d)
  pop!(ω, i)
  pop!(ω, j)
  k ∈ ω && pop!(ω, k)
  l ∈ ω && pop!(ω, l)
  return (k, l, [draw!(ω) for ~=3:N]...)
end

function twobody_test(i,j,k,l)
  @assert i<j && k<l
  s = draw_state(i,j,k,l)
  r = [ [ (k in (1,d+1) ? 1 : rand(0:5)) for n in QNTensorTrains.occupation_qn(N,d,k)] for k=1:d+1]

  ref_s = (i, j, s[3:N]...)

  p = round(tt_randn(Val(d),Val(N),r)); rmul!(p, 1/norm(p))
  Skl = delete!(delete!(Set(1:d),k),l)
  x0 = tt_state(s, Val(d)) + AdagᵢAⱼ(p, rand(Skl), k) + AdagᵢAⱼ(p, rand(Skl), l) # Random perturbation that should go to zero
  x = AdagᵢAdagⱼAₖAₗ(x0, i,j,k,l)

  QNTensorTrains.check(x)

  # Count electrons on each site
  nx = [dot(x, AdagᵢAⱼ(x, k, k)) / dot(x,x) for k=1:d]
  nref = [(k ∈ ref_s ? 1.0 : 0.0) for k=1:d]
 
  test = isapprox(norm(nx-nref),  0;  atol=10eps()*norm(x0))

  if !(test)
    @show i,j,k,l
    @show nx, nref, norm(nx-nref)/norm(x0)
  end

  return test
end

ω = Set(1:d)
S = sort([draw!(ω) for ~ = 1:4])

for i in S, j in filter(>(i), S), k in S, l in filter(>(k), S)
  @test twobody_test(i,j,k,l)
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

  x0 = tt_state(s1, Val(d)) + tt_state(s2, Val(d)) + tt_state(s3, Val(d))

  x = AdagᵢAⱼ( AdagᵢAⱼ(x0, 2,1), 4,4)

  ref = tt_state((2,4), Val(d))
  norm(x - ref, :LR) ≈ 0
end
