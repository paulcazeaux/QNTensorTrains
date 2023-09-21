using PyCall
using LinearAlgebra
using QNTensorTrains

pyscf = pyimport("pyscf")
fci = pyimport("pyscf.fci")

#
# Create Simple Molecule
#
# mol = pyscf.gto.M(atom = "N 0 0 0; N 0 0 1;", basis = "sto3g", verbose = 3)
# mol = pyscf.gto.M(atom = "o 0 0 0; o 0 0 1;", basis = "sto3g", verbose = 3)
mol = pyscf.gto.M(; atom="""C      0.00000    0.00000    0.00000
  H      0.00000    0.00000    1.08900
  H      1.02672    0.00000   -0.36300
  H     -0.51336   -0.88916   -0.36300
  H     -0.51336    0.88916   -0.36300""", basis="sto3g", verbose=3)

# Run HF
mf = pyscf.scf.RHF(mol).run()
println("RHF Energy (Ha): ", mf.e_tot)

# Create shorthands for 1- and 2-body integrals in MO basis
mo = mf.mo_coeff
n = size(mo, 1)
one_body = mo' * mf.get_hcore() * mo
two_body = reshape(mol.ao2mo(mf.mo_coeff; aosym=1), n, n, n, n)

# FCI (i.e. exact diagonalization)
cisolver = fci.FCI(mf)
cisolver.kernel()
println("FCI Energy (Ha): ", cisolver.e_tot)

#
# Setup for MPS Calculation
#
t = one_body
V = -.5 * permutedims(two_body, (3, 2, 1, 4))

↑(i) = 2i-1
↓(i) = 2i
d = 2n

t = zeros(d,d)
v = zeros(d,d,d,d)
for i=1:n, j=1:n
	t[↑(i),↑(j)] = one_body[i,j]
	t[↓(i),↓(j)] = one_body[i,j]
end
for i=1:n,j=1:n,k=1:n,l=1:n
    v[↑(j),↑(i),↑(k),↑(l)] = .5*two_body[j,k,i,l]
    v[↓(j),↓(i),↓(k),↓(l)] = .5*two_body[j,k,i,l]
    v[↓(j),↑(i),↓(k),↑(l)] = .5*two_body[j,k,i,l]
    v[↑(j),↓(i),↑(k),↓(l)] = .5*two_body[j,k,i,l]
end
v[abs.(v).<1e-8] .= 0

E(ψ) = RayleighQuotient(ψ,t,v) + e_nuclear

s = Vector{Bool}(undef, d)
for i=1:n
    ρ = mf.mo_occ[i]
    s[↑(i)] = s[↓(i)] = (ρ ≈ 0 ? false : (ρ ≈ 2 ? true : error("Occupation at $n is $ρ")))
end
e_nuclear = mf.energy_nuc()
ψmf = tt_state(s)
emf = E(ψmf)
@show E(ψmf)
println("Energy Error from MF MPS (Ha) ", abs(emf - mf.e_tot))

@show mf.e_tot - e_nuclear, cisolver.e_tot - e_nuclear
ψ0 = perturbation(ψmf, 10, 1e12)

@time e, ψ = MALS(t,v,ψ0)
display(ψ)
@show E(ψ)
println("DMRG Error ", abs(e+e_nuclear - cisolver.e_tot))
