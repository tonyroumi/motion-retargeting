import torch
import torch.nn as nn

# Simple model: linear layer
model = nn.Linear(3, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Dummy input and target
x = torch.randn(4, 3)
y = torch.randn(4, 1)

# Forward pass
pred = model(x)
loss = nn.functional.mse_loss(pred, y)

# Backward pass + optimizer step
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"loss: {loss.item():.4f}")
