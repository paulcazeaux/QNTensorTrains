"""
    QNTensorTrains

A Julia package collecting algorithms to create and manipulate tensors 
in the Quantum Number Preserving, Block-Sparse Tensor Train format 
conserving total fermionic particle number.
"""
module QNTensorTrains

using LinearAlgebra
import Base: @propagate_inbounds

include("base/util.jl")
include("base/block.jl")
include("base/sparsecore.jl")
include("base/unsafesparsecore.jl")
include("base/tensor.jl")
include("base/round.jl")
include("base/multiplication.jl")
include("base/addition.jl")
include("base/dot.jl")
include("base/complex.jl")

export TTvector, SparseCore, core, ⊕
export tt_zeros, tt_ones, tt_state
export move_core!, round!, round_global!
export times, power, lognorm
include("base/tangent.jl")
export TTtangent, component, retract, transport

include("base/matrixfree.jl")
include("base/hamiltonian.jl")
export Adag_view, A_view, AdagA_view, S_view, Id_view
export AdagᵢAⱼ_view, AdagᵢAdagⱼAₖAₗ_view

export Adag, A, AdagA, S, Id
export AdagᵢAⱼ, AdagᵢAdagⱼAₖAₗ

export Adag!, A!, AdagA!, S!, Id!
export AdagᵢAⱼ!, AdagᵢAdagⱼAₖAₗ!

using .Hamiltonian
export sparse_H_matvec, H_matvec, RayleighQuotient

include("random/rand.jl")
export tt_rand,tt_randn, perturbation

include("random/randomround.jl")
export roundRandOrth, roundRandOrth!

include("solvers/ALS.jl")
include("solvers/MALS.jl")
export ALS, MALS

end # module QNTensorTrains
