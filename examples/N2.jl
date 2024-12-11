using PythonCall
using LinearAlgebra
using QNTensorTrains

pyscf = pyimport("pyscf")
fci = pyimport("pyscf.fci")

basis="sto3g"

#
# Create Simple Molecule
#
mol = pyscf.gto.M(atom = "N 0 0 0; N 0 0 2.118;", basis = basis, unit="B", verbose = 3)

# Run HF
mf = pyscf.scf.RHF(mol).run()
println("RHF Energy (Ha): ", mf.e_tot)

# Create shorthands for 1- and 2-body integrals in MO basis
mo = pyconvert(Array, mf.mo_coeff)
hcore_ao = pyconvert(Array, mf.get_hcore())
d = size(mo, 1)
one_body = mo' * hcore_ao * mo
two_body = reshape(pyconvert(Array, mol.ao2mo(mf.mo_coeff; aosym=1)), d, d, d, d)

if basis=="ccpvdz"
    # FCI (i.e. exact diagonalization) from literature
    e_tot = −109.2821727
elseif basis=="sto3g"
    # # FCI (i.e. exact diagonalization)
    cisolver = fci.FCI(mf)
    cisolver.kernel()
    println("FCI Energy (Ha): ", cisolver.e_tot)
    e_tot = pyconvert(Float64, cisolver.e_tot)
else
    error()
end

# println("DMRG Error (Ha): ", dmrg_energy - e_tot)

#
# Setup for MPS Calculation



# Threads.nthreads()>1 && BLAS.set_num_threads(6)

s = Vector{Bool}(undef, d)
mo_occ = pyconvert(Array, mf.mo_occ)
for i=1:d
    ρ = mo_occ[i]
    s[i] = (ρ ≈ 0 ? false : (ρ ≈ 2 ? true : error("Occupation at $n is $ρ")))
end
e_nuclear = pyconvert(Float64, mf.energy_nuc())
ψmf = tt_state(s, s)
println("Building Hamiltonian MPO...")
@time H = SparseHamiltonian(one_body,two_body,ψmf)

E(ψ) = RayleighQuotient(H, ψ) + e_nuclear
emf = E(ψmf)
println("Energy Error from MF MPS (Ha) ", abs(emf - mf.e_tot))
println("Energy difference between MF MPS and FCI solution (HA) ", mf.e_tot - e_tot)
println()

@time e, ψ = MALS(H, perturbation(ψmf, 5, .01), ε=1e-4, maxIter=10, verbosity=true)
display(ψ)
@show E(ψ)-e_tot

@time e1, ψ1, _, hist1, res1 = randLanczos(H,ψmf; tol=1e-4, maxIter=20, rmax=10, over=10)
E1 = E(ψ1)
display(ψ1)
@show hist1.+e_nuclear.-e_tot, E1-e_tot
@time e2, ψ2, _, hist2, res2 = randLanczos(H,ψ1; tol=1e-6, maxIter=20, rmax=20, over=10)
E2 = E(ψ2)
display(ψ2)
@show hist2.+e_nuclear.-e_tot, E2-e_tot
@time e3, ψ3, _, hist3, res3 = randLanczos(H,ψ2; tol=1e-6, maxIter=20, rmax=30, over=10)
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
scatter!(abs.(hist1[1:end-1] .- (e_tot - e_nuclear)), label="rmax = 10")

N = length(res1).+(1:length(res2))
scatter!(N, abs.(hist2[1:end-1] .- (e_tot - e_nuclear)), label="rmax = 20")

N = (length(res1)+length(res2)).+(1:length(res3))
scatter!(N, abs.(hist3[1:end-1] .- (e_tot - e_nuclear)), label="rmax = 30")
xlabel!("Iterations")
ylabel!("Error (Ha)")
savefig("N2_$(basis).pdf")