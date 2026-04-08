import torch
import torch.nn as nn
import torch.optim as optim
from engine import MobileFaceLandmarker

def train_landmarker(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileFaceLandmarker().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(100):
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "landmarker.pth")

if __name__ == "__main__":
    print("Training script ready. Prepare your dataset (e.g. 300-W or WFLW).")
