using PyCall
using LinearAlgebra
using QNTensorTrains

pyscf = pyimport("pyscf")
fci = pyimport("pyscf.fci")

#
# Create Simple Molecule
#
mol = pyscf.gto.M(atom = "N 0 0 0; N 0 0 2.118;", basis = "ccpvdz", unit="B", verbose = 3)
# mol = pyscf.gto.M(atom = "o 0 0 0; o 0 0 1;", basis = "sto3g", verbose = 3)
# mol = pyscf.gto.M(; atom="""C      0.00000    0.00000    0.00000
#   H      0.00000    0.00000    1.08900
#   H      1.02672    0.00000   -0.36300
#   H     -0.51336   -0.88916   -0.36300
#   H     -0.51336    0.88916   -0.36300""", basis="sto3g", verbose=3)

# Run HF
mf = pyscf.scf.RHF(mol).run()
println("RHF Energy (Ha): ", mf.e_tot)

# Create shorthands for 1- and 2-body integrals in MO basis
mo = mf.mo_coeff
n = size(mo, 1)
one_body = mo' * mf.get_hcore() * mo
two_body = reshape(mol.ao2mo(mf.mo_coeff; aosym=1), n, n, n, n)

# # FCI (i.e. exact diagonalization)
# cisolver = fci.FCI(mf)
# cisolver.kernel()
# println("FCI Energy (Ha): ", cisolver.e_tot)
# e_tot = cisolver.e_tot
e_tot = −109.2821727

#
# Setup for MPS Calculation


↑(i) = 2i-1
↓(i) = 2i
d = 2n

t = zeros(d,d)
v = zeros(d,d,d,d)
for i=1:n, j=1:n
	t[↑(i),↑(j)] = one_body[i,j]
	t[↓(i),↓(j)] = one_body[i,j]
end
# Mindful of chemists' notation index ordering
for i=1:n,j=1:n,k=1:n,l=1:n
    v[↑(i),↑(j),↑(k),↑(l)] = 1/2 * two_body[i,k,j,l]
    v[↓(i),↓(j),↓(k),↓(l)] = 1/2 * two_body[i,k,j,l]
    v[↓(i),↑(j),↓(k),↑(l)] = 1/2 * two_body[i,k,j,l]
    v[↑(i),↓(j),↑(k),↓(l)] = 1/2 * two_body[i,k,j,l]
end
v[abs.(v).<1e-8] .= 0
# Reduce two-electron term - condense to i<j and k<l terms
v = QNTensorTrains.Hamiltonian.reduce(v)

E(ψ) = RayleighQuotient(ψ,t,v; reduced=true) + e_nuclear

s = Vector{Bool}(undef, d)
for i=1:n
    ρ = mf.mo_occ[i]
    s[↑(i)] = s[↓(i)] = (ρ ≈ 0 ? false : (ρ ≈ 2 ? true : error("Occupation at $n is $ρ")))
end
e_nuclear = mf.energy_nuc()
ψmf = tt_state(s)
emf = E(ψmf)
println("Energy Error from MF MPS (Ha) ", abs(emf - mf.e_tot))
println("Energy difference between MF MPS and FCI solution (HA) ", mf.e_tot - e_tot)
println()
# ψ0 = perturbation(ψmf, 10, .1)

# @time e, ψ = MALS(t,v,ψ0; reduced=true)
# display(ψ)
# @show E(ψ)
# println("DMRG Error ", abs(e+e_nuclear - e_tot))

@time e1, ψ1, hist1, res1 = randLanczos(t,v,ψmf; tol=1e-6, maxIter=10, rmax=50, reduced=true)
display(ψ1)
@show E(ψ1)
@time e2, ψ2, hist2, res2 = randLanczos(t,v,ψ1; tol=1e-6, maxIter=10, rmax=50, reduced=true)
display(ψ2)
@show E(ψ2)
@time e3, ψ3, hist3, res3 = randLanczos(t,v,ψ2; tol=1e-6, maxIter=20, rmax=100, reduced=true)
display(ψ3)
@show E(ψ3)
println("Lanczos Error ", abs(e3+e_nuclear - e_tot))

using Plots
plot(res1, label="residual")
plot!(abs.(hist1 .- (e_tot - e_nuclear)), yaxis=:log10, label="eigenvalue error")
plot!(res2, label="residual - restart")
plot!(abs.(hist2 .- (e_tot - e_nuclear)), yaxis=:log10, label="eigenvalue error - restart")

