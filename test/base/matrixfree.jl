# one body operator with i < j
@test begin
  T = Float64
  d = 6
  N = 2

  s1 = (2,4)
  s2 = (2,5)

  x0 = tt_state(s1, d) + tt_state(s2, d)

  x = deepcopy(x0); AdagᵢAⱼ!(x, 1,2)
  y = AdagᵢAⱼ(x0, 1,2)

  ref = (tt_state((1,4),d)+tt_state((1,5),d))
  norm(x-ref) + norm(y-ref) ≈ 0
end


# one body operator with i > j
@test begin
  T = Float64
  d = 6
  N = 2

  s1 = (6,4)
  s2 = (6,5)
  s3 = (2,3)

  x0 = tt_state(s1, d) + tt_state(s2, d) + tt_state(s3, d)
  x = deepcopy(x0); AdagᵢAⱼ!(x, 5,4)
  y = AdagᵢAⱼ(x0, 5,4)
  
  ref = tt_state((5,6), d)
  norm(x - ref) + norm(y - ref) ≈ 0
end

##############################
### Two body operator test ###
##############################

T = Float64
d = 10
N = rand(3:7)

# Helper draw function
function draw!(ω)
  i = rand(ω)
  pop!(ω, i)
  return i
end
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

  p = round(tt_randn(d,N,r)); rmul!(p, 1/norm(p))
  Skl = delete!(delete!(Set(1:d),k),l)
  x0 = tt_state(s, d) + AdagᵢAⱼ(p, rand(Skl), k) + AdagᵢAⱼ(p, rand(Skl), l) # Random perturbation that should go to zero
  x = deepcopy(x0); AdagᵢAdagⱼAₖAₗ!(x, i,j,k,l)
  y = AdagᵢAdagⱼAₖAₗ(x0, i,j,k,l)

  QNTensorTrains.check(x)
  QNTensorTrains.check(y)

  # Count electrons on each site
  nx = [dot(x, AdagᵢAⱼ(x, k, k)) / dot(x,x) for k=1:d]
  ny = [dot(y, AdagᵢAⱼ(y, k, k)) / dot(y,y) for k=1:d]
  nref = [(k ∈ ref_s ? 1.0 : 0.0) for k=1:d]
 
  test = isapprox(norm(x-y, :LR), 0; atol=10eps()*norm(x0)) && 
         isapprox(norm(nx-nref),  0;  atol=10eps()*norm(x0)) && 
         isapprox(norm(ny-nref),  0;  atol=10eps()*norm(x0))

  if !(test)
    @show i,j,k,l
    @show norm(x-y, :LR)
    @show nx, nref, norm(nx-nref)/norm(x0)
    @show ny, nref, norm(ny-nref)/norm(x0)
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

  x0 = tt_state(s1, d) + tt_state(s2, d) + tt_state(s3, d)

  x = deepcopy(x0); AdagᵢAⱼ!(x, 2,1); AdagᵢAⱼ!(x, 4,4)
  y = AdagᵢAⱼ( AdagᵢAⱼ(x0, 2,1), 4,4)

  ref = tt_state((2,4), d)
  norm(x - ref, :LR) + norm(y - ref, :LR) ≈ 0
end

# General creation operator
T = Float64
d = 10
N = 6

function normdiff(C1::SparseCore{T,N,d}, C2::SparseCore{T,N,d}) where {T<:Number,N,d}
  @assert C1.k == C2.k
  r = T(0)
  for n in axes(C1.unoccupied,1)
    if size(C1.unoccupied[n]) == size(C2.unoccupied[n])
      r += norm(Array(C2.unoccupied[n])-Array(C1.unoccupied[n]))
    else
      r += norm(Array(C1.unoccupied[n]))+norm(Array(C2.unoccupied[n]))
    end
  end
  for n in axes(C1.occupied,1)
    if size(C1.occupied[n]) == size(C2.occupied[n])
      r += norm(Array(C2.occupied[n])-Array(C1.occupied[n]))
    else
      r += norm(Array(C1.occupied[n]))+norm(Array(C2.occupied[n]))
    end
  end
  return r
end

for k = 1:d-1
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(0:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(0:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.Adag_left!(C1)
    C2 = deepcopy(C); QNTensorTrains.Adag!(C2, 0, 0, 1)
    C3,  = QNTensorTrains.Adag(C, 0, 0, 1)

    normdiff(C1, C2) + normdiff(C1, C3) ≈ 0
  end
end

for k = 2:d
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(0:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(0:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.Adag_right!(C1)
    C2 = deepcopy(C); QNTensorTrains.Adag!(C2, -1, 0, 1)
    C3,  = QNTensorTrains.Adag(C, -1, 0, 1)

    normdiff(C1, C2) + normdiff(C1, C3) ≈ 0
  end
end


# Two body annihilation operator
T = Float64
d = 10
N = 6

# Left
for k = 1:d-1
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.A_left!(C1)
    C2 = deepcopy(C); QNTensorTrains.A!(C2, 0, 0, 1)
    C3, = QNTensorTrains.A(C, 0, 0, 1)
    
    normdiff(C1,C2) + normdiff(C1,C3) ≈ 0
  end
end

# Right
for k = 2:d
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.A_right!(C1)
    C2 = deepcopy(C); QNTensorTrains.A!(C2, 1, 1, 0)
    C3, = QNTensorTrains.A(C, 1, 1, 0)
    
    normdiff(C1,C2) + normdiff(C1,C3) ≈ 0
  end
end


# Annihilation/Creation (local particle number) operator
T = Float64
d = 10
N = 6

for k = 1:d
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.AdagA!(C1)
    C2 = deepcopy(C); QNTensorTrains.AdagA!(C2, 0, 0, 1)
    C3 = QNTensorTrains.AdagA(C, 0, 0, 1)[1]
    
    normdiff(C1,C2) + normdiff(C1,C3) ≈ 0
  end
end

# Jordan-Wigner component
# Left
for k = 2:d-1
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.S⁺!(C1)
    C2 = deepcopy(C); QNTensorTrains.S!(C2, 1, 1, 0)
    C3, = QNTensorTrains.S(C, 1, 1, 0)
    
    normdiff(C1,C2) + normdiff(C1,C3) ≈ 0
  end
end

# Right
for k = 2:d-1
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.S⁻!(C1)
    C2 = deepcopy(C); QNTensorTrains.S!(C2, -1, 0, 1)
    C3, = QNTensorTrains.S(C, -1, 0, 1)
    
    normdiff(C1,C2) + normdiff(C1,C3) ≈ 0
  end
end


# Identity component
# Left
for k = 1:d-1
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.I_left!(C1)
    C2 = deepcopy(C); QNTensorTrains.Id!(C2, 0, 0, 1)
    C3, = QNTensorTrains.Id(C, 0, 0, 1)
    
    normdiff(C1,C2) + normdiff(C1,C3) ≈ 0
  end
end

# Right
for k = 2:d
  @test begin
    C = randn(N,d,k, [k == 1 ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k  )], 
                     [k == d ? 1 : rand(1:6) for n in QNTensorTrains.occupation_qn(N,d,k+1)],
          T)

    C1 = deepcopy(C); QNTensorTrains.OneBody.I_right!(C1)
    C2 = deepcopy(C); QNTensorTrains.Id!(C2, 0, 1, 0)
    C3, = QNTensorTrains.Id(C, 0, 1, 0)
    
    normdiff(C1,C2) + normdiff(C1,C3) ≈ 0
  end
end