import torch

w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
x_data = torch.tensor([5.0, 2.0, 4.0, 7.0, -4.0, -1.0])
y_data = torch.tensor([11.0, 5.0, 9.0, 15.0, -7.0, -1.0])

for epoch in range(150):
    loss = ((w * x_data + b - y_data) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
    w.grad.zero_()
    b.grad.zero_()


print(f"y={w.item():.2f}x + {b.item():.2f}")
