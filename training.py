import torch
import torch.nn as nn
import torch.optim as optim
from engine import PNet


class MTCNNLoss(nn.Module):
    """Multi-task Loss for classification and bbox regression."""
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.MSELoss()

    def forward(self, pred_cls, pred_box, target_cls, target_box):
        # Mask only 'face' samples for bounding box regression
        mask = target_cls == 1
        c_loss = self.cls_loss(pred_cls, target_cls)
        b_loss = self.box_loss(pred_box[mask], target_box[mask]) if mask.any() else 0
        return c_loss + 0.5 * b_loss


def train_model(dataloader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = MTCNNLoss()

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, label, bbox in dataloader:
            img, label, bbox = img.to(device), label.to(device), bbox.to(device)

            optimizer.zero_grad()
            out_cls, out_bbox = model(img)
            
            # Squeeze output to match (Batch, Classes) and (Batch, Coords)
            loss = criterion(out_cls.squeeze(), out_bbox.squeeze(), label, bbox)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "pnet_final.pth")
