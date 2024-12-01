"""
    QNTensorTrains

A Julia package collecting algorithms to create and manipulate tensors 
in the Quantum Number Preserving, Block-Sparse Tensor Train format 
conserving total fermionic particle number.
"""
module QNTensorTrains
using OffsetArrays, LinearAlgebra, VectorInterface, Random, TimerOutputs
import Base: @propagate_inbounds

abstract type AbstractCore{T<:Number,Nup,Ndn,d} <: AbstractArray{T,3} end
abstract type AbstractAdjointCore{T<:Number,Nup,Ndn,d} <: AbstractArray{T,3} end
abstract type AbstractFrame{T<:Number,Nup,Ndn,d} <: AbstractArray{T,2} end
abstract type FrameView{T,Nup,Ndn,d} <: AbstractFrame{T,Nup,Ndn,d} end
abstract type SparseCoreView{T,Nup,Ndn,d} <: AbstractCore{T,Nup,Ndn,d} end

@enum Spin Up=1 Dn=-1
const Orbital = @NamedTuple{site::Int, spin::Spin}

include("base/frame.jl")
include("base/util.jl")
include("base/sparsecore.jl")
include("base/vector.jl")
include("base/round.jl")
include("base/multiplication.jl")
include("base/addition.jl")
include("base/dot.jl")
include("base/complex.jl")

export TTvector, Frame, SparseCore, core, ⊕
export tt_zeros, tt_ones, tt_state
export move_core!, round!, round_global!
export times, power, lognorm, roundSum
export row_ranks, col_ranks
include("base/tangent.jl")
export TTtangent, component, retract, transport

include("base/elementaryops.jl")
include("base/hamiltonian.jl")

export Adag, A, AdagA, S, Id
export AdagᵢAⱼ, AdagᵢAdagₖAₗAⱼ

using .Hamiltonian
export SparseHamiltonian, H_matvec_core, H_matvec, RayleighQuotient, xᵀHy

include("random/rand.jl")
include("random/round.jl")
include("random/sum.jl")
include("random/matvec.jl")
export tt_rand,tt_randn,tt_randd, perturbation
export roundRandOrth, roundRandOrth!
export roundRandSum
export randRound_H_MatVec, randRound_H_MatVec!
export roundRandSum2
export roundRandOrth2, roundRandOrth2!
export randRound_H_MatVec2, randRound_H_MatVec2!

include("solvers/ALS.jl")
include("solvers/MALS.jl")
include("solvers/lanczos.jl")
export ALS, MALS, Lanczos, randLanczos

end # module QNTensorTrains
