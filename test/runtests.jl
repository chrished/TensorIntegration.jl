using TensorIntegration
using Test


# add test for basic integrals
N = 5
# uniform expectation 
a = 1. 
b = 3. 
pdf(x, a, b) = 1./(b-a)
grid, weights = TensorIntegration.simpson(a, b, N)

Exp = (grid.*pdf.(grid, a, b))'*weights
@test Exp == (a+b)/2.


# now in 2d
N2d = [5, 5]
a2d = [1., 1.]
b2d = [2., 2.]
pdf(x, a, b, d) = 1/(b[d]-a[d])

grid, weights = TensorIntegration.tensor_simpson(a2d, b2d, N2d)
Exp1 = (grid[:,1].*pdf(grid, a2d, b2d, 1))'*weights
Exp2 = (grid[:,2].*pdf(grid, a2d, b2d, 2))'*weights

@test Exp1[1] == (a2d[1]+b2d[1])/2.
@test Exp2[1] == (a2d[2]+b2d[2])/2.


# now in 3d
N3d = [5, 5, 5]
a3d = [1., 1., 0.1]
b3d = [2., 2., 3.3]
pdf(x, a::Array, b::Array) = 1./prod(b3d.-a3d)

grid, weights = TensorIntegration.tensor_simpson(a3d, b3d, N3d)
Exp1 = (grid[:,1].*pdf(grid, a3d, b3d))'*weights
Exp2 = (grid[:,2].*pdf(grid, a3d, b3d))'*weights
Exp3 = (grid[:,3].*pdf(grid, a3d, b3d))'*weights

@test isapprox(Exp1[1], (a3d[1]+b3d[1])/2.)
@test isapprox(Exp2[1], (a3d[2]+b3d[2])/2.)
@test isapprox(Exp3[1], (a3d[3]+b3d[3])/2.)

# now with one dimension only 1 point 
N3d = [5, 5, 1]
a3d = [1., 1., 1.]
b3d = [2., 2., 2.]
pdf(x, a::Array, b::Array) = 1./prod(b3d.-a3d)

grid, weights = TensorIntegration.tensor_simpson(a3d, b3d, N3d)
Exp1 = (grid[:,1].*pdf(grid, a3d, b3d))'*weights
Exp2 = (grid[:,2].*pdf(grid, a3d, b3d))'*weights
Exp3 = (grid[:,3].*pdf(grid, a3d, b3d))'*weights

@test isapprox(Exp1[1], (a3d[1]+b3d[1])/2.)
@test isapprox(Exp2[1], (a3d[2]+b3d[2])/2.)
@test isapprox(Exp3[1], (a3d[3]+b3d[3])/2.)