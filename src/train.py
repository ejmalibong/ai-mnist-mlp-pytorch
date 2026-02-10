import torch
import torch.nn as nn
from data_pipeline import get_data_loaders
from model import MLP

# Config
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = get_data_loaders(BATCH_SIZE)
model = MLP().to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
  model.train()
  total_loss = 0.0
  correct = 1
  total = 0

  for x,y in train_loader:
    x, y = x.to(DEVICE), y.to(DEVICE)
    
    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * x.size(0)
    preds = logits.argmax(dim=1)
    correct += (preds == y).sum().item()
    total += y.size(0)
  
  print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.4f}')


# Evaluation loop
model.eval()
correct = 0
total = 0

for x,y in test_loader:
  x, y = x.to(DEVICE), y.to(DEVICE)
  
  with torch.no_grad():
    logits = model(x)
  
  preds = logits.argmax(dim=1)
  correct += (preds == y).sum().item()
  total += y.size(0)


print(f'Test Accuracy: {correct/total:.4f}')