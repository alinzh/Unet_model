import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch import nn
from torchvision import transforms
from model import UNet
import numpy as np
from torchmetrics.classification import JaccardIndex

num_epochs = 1
num_classes = 37
batch_size = 10
learning_rate = 0.001


class MaskToTensor:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, mask):
        mask = np.array(mask)
        mask_tensor = torch.zeros((self.num_classes, mask.shape[0], mask.shape[1]), dtype=torch.float32)

        for class_idx in range(self.num_classes):
            mask_tensor[class_idx] = torch.from_numpy((mask == class_idx).astype(np.int64))

        return mask_tensor


desired_size = (256, 256)
num_classes = 37

# Создание новых датасетов с применением своих трансформаций
training_data = datasets.OxfordIIITPet(
    root="data",
    download=True,
    target_types="segmentation",
    transform=transforms.Compose([
        transforms.Resize(desired_size),
        transforms.ToTensor(),
    ]),
    target_transform=transforms.Compose([
        transforms.Resize(desired_size),
        MaskToTensor(num_classes),
    ]),
    split="trainval"
)

resized_test_data = datasets.OxfordIIITPet(
    root="data",
    download=True,
    target_types="segmentation",
    transform=transforms.Compose([
        transforms.Resize(desired_size),
        transforms.ToTensor(),
    ]),
    target_transform=transforms.Compose([
        transforms.Resize(desired_size),
        MaskToTensor(num_classes),
    ]),
    split="test"
)

# Создание DataLoader с новыми датасетами
train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=resized_test_data, batch_size=batch_size, shuffle=False)


model = UNet(37)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

totalTrainLoss = 0
totalTestLoss = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        print(i, "/", total_step)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()
        if i > 10:
            break

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss/total_step:.4f}")

total_iou = 0
jaccard = JaccardIndex(task="multiclass", num_classes=2)

model.eval()
k = 0
with torch.no_grad():
    for images, mask in test_loader:
        k += 1
        outputs = model(images)
        total_iou += jaccard(outputs, mask).cpu()
        if k > 10:
            break

    mean_iou = total_iou / len(test_loader)
    print('Test Mean Intersection over Union (IoU) of the model on the test images: {:.2f}%'.format(mean_iou * 100))

# Сохраняем модель и строим график
torch.save(model.state_dict(), 'model_weights.pth' + 'conv_net_model.ckpt')