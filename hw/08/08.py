import torch

x = torch.tensor(5., requires_grad = True)
y = torch.tensor(5., requires_grad = True)
z = torch.tensor(5., requires_grad = True)

lr = 0.01

for i in range(1000):
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8
    f.backward()
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
        z -= lr * z.grad
        x.grad.zero_()
        y.grad.zero_()
        z.grad.zero_()

print(f"x= {x} y= {y} z= {z}")
# 1 2 3 = -6