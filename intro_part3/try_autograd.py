from data_loader import *

x = torch.randn(2,2, requires_grad=True)
y = x**2

print(y.grad_fn)

z = y.mean()
print(z)
print(x.grad)

z.backward()
print(x.grad)
print(x/2)