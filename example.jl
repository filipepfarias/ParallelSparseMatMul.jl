using Distributed
using Test
addprocs(3)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using SharedArrays
@everywhere using ParallelSparseMatMul
@everywhere using BenchmarkTools
@everywhere import Statistics:median
#  construct a trivial example with a triangular upper matrix
#n = 100
n =3*10^5
a = shsprand(n,n, 0.01);
x = rand(n)
xs = share(x)
ys = share(zeros(n))
val1, t1 = @timed A_mul_B!(ys, a, xs)
b = SparseMatrixCSC(a)
#y = b * x
val2, t2 = @timed y = b * x

# At_mul_B!
val3 , t3 = @timed At_mul_B!(ys, a, xs);
y4, t4  = @timed y = b' * x;
using Printf
println("|  op  | N           |  sparse      | normal         |");
println("|------|-------------|--------------|----------------|");
s1 = @sprintf "| A*x  | %.3e   |  %.4e  |   %.4e   | " n t1 t2;
s2 = @sprintf "| A'*x | %.3e   |  %.4e  |   %.4e   | " n t3 t4;
println(s1);
println(s2);

