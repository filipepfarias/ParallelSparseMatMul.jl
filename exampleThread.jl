using Test
using Pkg
Pkg.activate(".")
using ParallelSparseMatMul
using BenchmarkTools
import Statistics:median
using SparseArrays
#  construct a trivial example with a triangular upper matrix
#n = 100
n =3*10^3
a = sprand(n,n, 0.01);
x = rand(n)
y1 = copy(x)
y2 = copy(x)
y3 = zeros(size(x))
y4 = zeros(size(x))
val1, t1 = @timed At_mul_B!(y1, a, x)
val2, t2 = @timed y2 = a' * x;
@test y1 ≈ y2 atol=1e-12
using LinearAlgebra
# At_mul_B!
val3, t3 = @timed A_mul_B!(y3, a, x);
val4, t4  = @timed y4 = a * x;
@test y3 ≈ y4 atol=1e-12
using Printf
println("|  op  | N           |  sparse      | normal         |");
println("|------|-------------|--------------|----------------|");
s1 = @sprintf "| A'*x | %.3e   |  %.4e  |   %.4e   | " n t1 t2;
s2 = @sprintf "| A*x  | %.3e   |  %.4e  |   %.4e   | " n t3 t4;
println(s1);
println(s2);
