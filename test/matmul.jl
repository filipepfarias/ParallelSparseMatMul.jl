#  construct a trivial example with a triangular upper matrix
n = 10
a = share(reshape([(i <= j)* j for j=1:n for i=1:n], n, n))
y = SharedArray{Float64}(n)
x = share([1. * i for i=1:n])
y = a*x
yₜₕ = share([sum((k:n).^2) for k=1:n])
@test yₜₕ ≈ y atol=1e-10
