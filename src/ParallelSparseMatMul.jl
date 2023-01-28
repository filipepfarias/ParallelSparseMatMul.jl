module ParallelSparseMatMul

using Distributed
using SharedArrays
using SparseArrays
import Base: ==, adjoint, *, size, display, getindex

export shspeye, shsprand, shsprandn, shmem_randsample, SparseMatrixCSC, SharedSparseMatrixCSC, share, display, sdata, operator, nfilled, size, A_mul_B!, At_mul_B!
#export getindex, getindex_cols, shspeye, shsprand, shsprandn, shmem_randsample, SharedBilinearOperator, SharedSparseMatrixCSC, share, display, sdata, operator, nfilled, size, A_mul_B

# package code goes here
include("parallel_matmul.jl")
include("indexing.jl")
include("initialization.jl")

end # module