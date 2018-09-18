using Distributed
addrocs(3) # run all tests with 4 procs
@everywhere using Test
@everywhere using SharedSparseMatMul
@everywhere using LinearAlgebra
@everywhere using SparseArrays
@everywhere using SharedArrays
#rootDir = dirname(dirname(Base.functionloc(FeKode.eval)[1])) # fix with version 1.0
myTests = [ "matmul" ]
@testset "SPSMatMul" begin
    for t in myTests
        include("$(t).jl")
    end
end
