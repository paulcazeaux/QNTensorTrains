using PythonCall
using LinearAlgebra
using QNTensorTrains

using ITensors
using ITensorChemistry

Bohr = 0.529177249
molecule = Molecule([Atom("N"),Atom("N",0,0,2.118*Bohr)])
basis = "sto-3g"
@show molecule

hf = molecular_orbital_hamiltonian(molecule; basis);
hamiltonian = hf.hamiltonian;
hartree_fock_state = hf.hartree_fock_state
hartree_fock_energy = hf.hartree_fock_energy

# qubit_hamiltonian = jordanwigner(hamiltonian);
# qubit_state = jordanwigner(hartree_fock_state)
# @show qubit_hamiltonian[end];
# println("Number of qubit operators = ", length(qubit_hamiltonian))
# println("Hartree-Fock state |HF⟩ = |", prod(string.(qubit_state)),"⟩")

# hilbert space
s = siteinds("Fermion", 2*length(hartree_fock_state); conserve_qns=true)
fermion_hamiltonian = ITensorChemistry.electron_to_fermion(hamiltonian)
H = MPO(fermion_hamiltonian, s)

# initialize MPS to HF state
fermion_hf_state = ITensorChemistry.electron_to_fermion(hartree_fock_state)
ψhf = MPS(s, fermion_hf_state)

# run dmrg
# dmrg_kwargs = (;
#   nsweeps=10,
#   maxdim=[100,200],
#   cutoff=1e-12,
#   noise=[1e-3, 1e-5, 1e-8, 0.0],
# )
# @time dmrg_energy, ψ = dmrg(H, ψhf; outputlevel=1, dmrg_kwargs...)
println("Hartree-Fock Energy: ", hartree_fock_energy)
# println("DMRG Energy: ", dmrg_energy)


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
n = size(mo, 1)
one_body = mo' * hcore_ao * mo
two_body = reshape(pyconvert(Array, mol.ao2mo(mf.mo_coeff; aosym=1)), n, n, n, n)

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

Threads.nthreads()>1 && BLAS.set_num_threads(1)

s = Vector{Bool}(undef, d)
mo_occ = pyconvert(Array, mf.mo_occ)
for i=1:n
    ρ = mo_occ[i]
    s[↑(i)] = s[↓(i)] = (ρ ≈ 0 ? false : (ρ ≈ 2 ? true : error("Occupation at $n is $ρ")))
end
e_nuclear = pyconvert(Float64, mf.energy_nuc())
ψmf = tt_state(s)

H = SparseHamiltonian(t,v,ψmf; ϵ=1e-6)
E(ψ) = RayleighQuotient(H, ψ) + e_nuclear

emf = E(ψmf)
println("Energy Error from MF MPS (Ha) ", abs(emf - mf.e_tot))
println("Energy difference between MF MPS and FCI solution (HA) ", mf.e_tot - e_tot)
println()


@time e1, ψ1, _, hist1, res1 = randLanczos(H,ψmf; tol=1e-4, maxIter=20, rmax=150, over=10, reduced=true)
E1 = E(ψ1)
display(ψ1)
@show hist1.+e_nuclear.-e_tot, E1-e_tot
@time e2, ψ2, _, hist2, res2 = randLanczos(H,ψ1; tol=1e-6, maxIter=20, rmax=150, over=10, reduced=true)
E2 = E(ψ2)
display(ψ2)
@show hist2.+e_nuclear.-e_tot, E2-e_tot
@time e3, ψ3, _, hist3, res3 = randLanczos(H,ψ2; tol=1e-6, maxIter=20, rmax=200, over=10, reduced=true)
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
xlabel!("Iterations")
ylabel!("Error (Ha)")
savefig("N2_$(basis).pdf")