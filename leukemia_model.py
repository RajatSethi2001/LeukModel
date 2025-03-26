import gc
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.mobilenetv2 import MobileNetV2

class LeukemiaDataset(Dataset):
    def __init__(self, all_filenames, hem_filenames):
        self.data = []
        for all_filename in all_filenames:
            self.data.append((all_filename, 1.0))
        
        for hem_filename in hem_filenames:
            self.data.append((hem_filename, 0.0))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(4608, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return torch.squeeze(x, 1)


class EffModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MobileNetV2(1)

    def forward(self, x):
        x = self.model(x)
        x = F.sigmoid(x)
        return x
        

def transform_image(filename, image_resize):
    image = Image.open(filename)
    image = image.resize(image_resize)
    transposes = [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]
    transpose1 = random.choice(transposes)
    transpose2 = random.choice(transposes)
    image = image.transpose(transpose1).transpose(transpose2)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))

    return image_array

def main():
    seed = 231
    random.seed(seed)
    np.random.seed(seed)
    model_savefile = "eff_leukemia.pth"
    batch_size = 256
    lr = 0.001
    epochs = 100
    lambda_reg = 0.01
    image_resize = (128, 128)
    model = EffModel()
    if os.path.exists(model_savefile):
        model.load_state_dict(torch.load(model_savefile, weights_only=True))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    all_filepath = "C-NMC_Leukemia/all/"
    hem_filepath = "C-NMC_Leukemia/hem/"
    all_filenames = [all_filepath + filename for filename in os.listdir(all_filepath)]
    hem_filenames = [hem_filepath + filename for filename in os.listdir(hem_filepath)]
    random.shuffle(all_filenames)
    random.shuffle(hem_filenames)

    total_images = min(len(all_filenames), len(hem_filenames))
    all_filenames = all_filenames[:total_images]
    hem_filenames = hem_filenames[:total_images]

    test_pct = 0.2
    train_test_bound = int(total_images * test_pct)

    train_all = all_filenames[train_test_bound:]
    train_hem = hem_filenames[train_test_bound:]

    test_all = all_filenames[:train_test_bound]
    test_hem = hem_filenames[:train_test_bound]

    train_dataset = LeukemiaDataset(train_all, train_hem)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = LeukemiaDataset(test_all, test_hem)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        correct_preds = 0
        total_preds = 0
        batch = 0
        epoch_loss = 0
        for inputs, labels in train_dataloader:
            pictures = []
            for filename in inputs:
                image_array = transform_image(filename, image_resize)
                pictures.append(image_array)
            
            optimizer.zero_grad()
            pictures = torch.tensor(np.array(pictures), dtype=torch.float32)
            labels = labels.to(torch.float32)
            outputs = model(pictures)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            batch_correct = torch.sum((torch.round(outputs) == labels).to(torch.float32)).item()
            correct_preds += batch_correct
            total_preds += len(labels)

            print(f"Training Batch = {batch + 1}, Batch Accuracy = {round(batch_correct / len(labels), 2)}, Batch Loss = {round(loss.item(), 4)}")
            batch += 1
        
            gc.collect()
        
        print(f"Epoch Training Accuracy = {round(correct_preds / total_preds, 2)}, Epoch Loss = {round(epoch_loss, 4)}")

        print("Saving Model")
        torch.save(model.state_dict(), model_savefile)
        print("Model Saved")

        if (epoch + 1) % 5 == 0:
            model.eval()
            correct_preds = 0
            total_preds = 0
            batch = 0
            for inputs, labels in test_dataloader:
                print(f"Testing Batch = {batch + 1}")
                pictures = []
                for filename in inputs:
                    image_array = transform_image(filename, image_resize)
                    pictures.append(image_array)
                
                pictures = torch.tensor(np.array(pictures), dtype=torch.float32)
                labels = labels.to(torch.float32)
                outputs = model(pictures)

                correct_preds += torch.sum((torch.round(outputs) == labels).to(torch.float32)).item()
                total_preds += len(labels)
                batch += 1

            print(f"Epoch Testing Accuracy = {round(correct_preds / total_preds, 2)}")
            model.train()

if __name__=="__main__":
    main()