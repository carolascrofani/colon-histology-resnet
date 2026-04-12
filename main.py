import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import shutil
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report


class IstologiaDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.Y[idx]

transform = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Carica il dataset LC25000 da Kaggle:
path = '/content/istologia/'
colon_n = os.listdir(f'{path}colon_n')
colon_aca = os.listdir(f'{path}colon_aca')
path_n = '/content/istologia/colon_n/'
path_aca = '/content/istologia/colon_aca/'

X = [f'{path_n}{i}' for i in colon_n] + [f'{path_aca}{x}' for x in colon_aca]
Y = [0 for _ in colon_n] + [1 for _ in colon_aca]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, train_size=0.5, random_state=42)

train_ds = IstologiaDataset(X_train,Y_train,transform=transform_train)
val_ds = IstologiaDataset(X_val, Y_val, transform=transform)
test_ds = IstologiaDataset(X_test, Y_test, transform=transform)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.fc = nn.Linear(2048, 2).to(device)
#model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(device)
#model.classifier[1] = nn.Linear(1280, 2).to(device)

'''
for param in model.parameters():
  param.requires_grad = False

model.fc.requires_grad_(True)
'''

losses = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_val_err = float('inf')
patient = 3
index = 0

#model.load_state_dict(torch.load('modello.pth'))


for epoch in range(10):
    model.train()
    err = 0
    for img, label in train_dl:
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = losses(output, label)
        loss.backward()
        optimizer.step()

        err = err+loss.item()
    err = err/len(train_dl)

    with torch.no_grad():
        model.eval()
        val_err = 0
        for img, label in val_dl:
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = losses(output, label)

            val_err = val_err + loss
        val_err = val_err/len(val_dl)
        if val_err < best_val_err:
            best_val_err = val_err
            index = 0
            torch.save(model.state_dict(), 'modello.pth')
        else:
            index += 1

        if index >= patient:
            break


    print(f'Epoch: {epoch}: errore: {err}, val_err: {val_err}')

model.eval()

with torch.no_grad():
    corrects = 0
    for img, label in test_dl:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        pred = torch.argmax(output, dim=1)
        corrects += (pred == label).sum().item()

    accuracy = corrects/len(test_ds)

print(f'accuracy: {accuracy}')


all_preds = []
all_labels = []
all_probs = []

model.eval()
with torch.no_grad():
    for img, label in test_dl:
        img = img.to(device)
        output = model(img)
        probs = torch.softmax(output, dim=1)[:, 1]
        preds = torch.argmax(output, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.numpy())
        all_probs.extend(probs.cpu().numpy())


fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('1 - Specificità (False Positive Rate)')
plt.ylabel('Sensibilità (True Positive Rate)')
plt.title('ROC Curve')
plt.legend()
plt.show()


#grad-cam
model.eval()

img_pil = Image.open(test_ds.X[0])
img_pil = img_pil.resize((224, 224))
img_np = np.array(img_pil).astype(np.float32) / 255.0


target_layers = [model.layer4[-1]]
img_tensor, label = test_ds[0]
img_tensor = img_tensor.unsqueeze(0).to(device)
cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(1)]
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
grayscale_cam = grayscale_cam[0]

visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_np)
axes[0].set_title(f'Originale - Label: {label}')
axes[0].axis('off')
axes[1].imshow(visualization)
axes[1].set_title('Grad-CAM')
axes[1].axis('off')
plt.show()
