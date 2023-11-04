using PyCall
using LinearAlgebra
using QNTensorTrains

pyscf = pyimport("pyscf")
fci = pyimport("pyscf.fci")

#
# Create Simple Molecule
#
# mol = pyscf.gto.M(atom = "N 0 0 0; N 0 0 2.118;", basis = "ccpvdz", unit="B", verbose = 3)
mol = pyscf.gto.M(atom = "N 0 0 0; N 0 0 2.118;", basis = "sto3g", unit="B", verbose = 3)

# Run HF
mf = pyscf.scf.RHF(mol).run()
println("RHF Energy (Ha): ", mf.e_tot)

# Create shorthands for 1- and 2-body integrals in MO basis
mo = mf.mo_coeff
n = size(mo, 1)
one_body = mo' * mf.get_hcore() * mo
two_body = reshape(mol.ao2mo(mf.mo_coeff; aosym=1), n, n, n, n)

# FCI (i.e. exact diagonalization) from literature
# e_tot = −109.2821727
# # FCI (i.e. exact diagonalization)
cisolver = fci.FCI(mf)
cisolver.kernel()
println("FCI Energy (Ha): ", cisolver.e_tot)
e_tot = cisolver.e_tot


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
# Reduce two-electron term - condense to i<j and k<l terms
v = QNTensorTrains.Hamiltonian.reduce(v, tol=1e-12)

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

@time e1, ψ1, hist1, res1 = randLanczos(t,v,ψmf; tol=1e-4, maxIter=20, rmax=50, over=15, reduced=true)
E1 = E(ψ1)
display(ψ1)
@show hist1.+e_nuclear.-e_tot, E1-e_tot
@time e2, ψ2, hist2, res2 = randLanczos(t,v,ψ1; tol=1e-6, maxIter=20, rmax=100, over=20, reduced=true)
E2 = E(ψ2)
display(ψ2)
@show hist2.+e_nuclear.-e_tot, E2-e_tot
@time e3, ψ3, hist3, res3 = randLanczos(t,v,ψ2; tol=1e-6, maxIter=20, rmax=150, over=20, reduced=true)
E3 = E(ψ3)
display(ψ3)
@show hist3.+e_nuclear.-e_tot, E3-e_tot
println("True Lanczos Residual ", abs(e3+e_nuclear - e_tot))

using Plots
plot(cumsum(length.([hist1, hist2, hist3]).-1), abs.(last.([hist1, hist2, hist3]).- (e_tot - e_nuclear)), 
            ls=:dot, seriestype=:scatter, yaxis=:log10,
            yticks=10.0 .^ (floor(log10(abs(e3+e_nuclear-e_tot))):ceil(log10(abs(emf-e_tot)))),
            label="Final eigenvalue error before restart",
            legend=:bottomleft)
scatter!([res1;res2;res3], label="residual")
scatter!(abs.(hist1[1:end-1] .- (e_tot - e_nuclear)), label="rmax = 50")

N = length(res1).+(1:length(res2))
scatter!(N, abs.(hist2[1:end-1] .- (e_tot - e_nuclear)), label="rmax = 100")

N = (length(res1)+length(res2)).+(1:length(res3))
scatter!(N, abs.(hist3[1:end-1] .- (e_tot - e_nuclear)), label="rmax = 150")

