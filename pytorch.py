import os
import sys
import xml.etree.ElementTree as ET
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import cv2
import numpy as np
from PIL import Image
import multiprocessing

multiprocessing.set_start_method("spawn")

# Define your dataset
class VOCDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "extracted_frames"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "extracted_frames", self.imgs[idx])
        label_path = os.path.join(self.root, "Annotations", self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        boxes, labels = [], []

        # Parse XML (if exists)
        if os.path.exists(label_path):
            tree = ET.parse(label_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # Example class label

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32).view(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target



    def __len__(self):
        return len(self.imgs)

# Transform
def get_transform():
    return ToTensor()

def collate_fn(batch):
    return tuple(zip(*batch))

# Initialize dataset and data loader
dataset = VOCDataset(root="CVAT_data", transforms=get_transform())
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

# Load and modify model
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Training setup
device = torch.device("mps")  # Use Metal (MPS)
for i, (image, target) in enumerate(data_loader):
    print(f"Batch {i}, Target: {target}")

# model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for images, targets in data_loader:
#         print(targets[0].items())
#         images = [img.to(device) for img in images]
#         targets = [
#             {
#                 "boxes": torch.tensor(t["boxes"], dtype=torch.float32).to(device),
#                 "labels": torch.tensor(t["labels"], dtype=torch.int64).to(device),
#             }
#             for t in targets
#         ]


#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         epoch_loss += losses.item()
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# # Save the model
# torch.save(model.state_dict(), "fasterrcnn_voc_model.pth")
# print("Training complete and model saved.")

