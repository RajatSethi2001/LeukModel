import gc
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import MobileNetV2

class BoneDataset(Dataset):
    def __init__(self, full_df, bone_type, train=True, train_test_split=0.2, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        self.full_df = full_df
        self.bone_type = bone_type
        self.train = train
        self.train_test_split = train_test_split
        self.seed = seed

        self.bone_df: pd.DataFrame = full_df[full_df[bone_type] == 1].reset_index(drop=True)
        self.bone_df = self.bone_df.iloc[np.random.permutation(len(self.bone_df))].reset_index(drop=True)

        self.columns = self.bone_df.columns.to_list()
        self.conditions = self.columns[22:31]

        rows = len(self.bone_df)
        row_split = int(train_test_split * rows)
        if train:
            self.bone_df = self.bone_df.iloc[row_split:, :].reset_index(drop=True)
        else:
            self.bone_df = self.bone_df.iloc[:row_split, :].reset_index(drop=True)

        self.augment_bone_df()

        self.bone_df = self.bone_df.iloc[np.random.permutation(len(self.bone_df))].reset_index(drop=True)

        # for condition in self.conditions:
        #     print(condition, len(self.bone_df[self.bone_df[condition] == 1]))
        
    def augment_bone_df(self):
        cancer_counts = []
        for condition in self.conditions:
            indices = []
            for index in self.bone_df.index:
                if self.bone_df.loc[index, condition]:
                    indices.append(index)
            cancer_counts.append([condition, indices])
        
        cancer_counts.sort(key = lambda x: len(x[1]), reverse=True)
        cancer_max = len(cancer_counts[0][1])

        for cancer_count in cancer_counts:
            cancer_name = cancer_count[0]
            cancer_rows = cancer_count[1]
            if len(cancer_rows) == 0:
                break
            augments_needed = cancer_max - len(cancer_rows)
            for augment in range(augments_needed):
                random_row = random.choice(cancer_rows)
                self.bone_df = pd.concat([self.bone_df, self.bone_df.loc[[random_row]]], ignore_index=True)

        self.bone_df = self.bone_df.reset_index(drop=True)

    def __len__(self):
        return len(self.bone_df)
    
    def __getitem__(self, index):
        filename = "images/" + self.bone_df.loc[index, "image_id"]
        labels = np.array(self.bone_df.loc[index, self.conditions], dtype=np.float32)
        return filename, labels

class BoneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MobileNetV2(9)

    def forward(self, x):
        x = F.softmax(self.model(x), dim=1)
        return x

def transform_image(filename):
    image = Image.open(filename).convert("RGB")
    image = image.resize((224, 224))
    image = image.rotate(random.randint(0, 359))
    # image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(image)
    # image = image.filter(ImageFilter.GaussianBlur(radius=2))
    # image = ImageEnhance.Sharpness(image).enhance(5.0)
    # image = ImageEnhance.Contrast(image).enhance(2)
    # image = image.filter(ImageFilter.FIND_EDGES)
    # image = image.filter(ImageFilter.EDGE_ENHANCE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return image_array

def main():
    dataset_df = pd.read_excel("dataset.xlsx")
    bone_type = "tibia"
    batch_size = 16
    epochs = 100
    lr = 0.001
    model_savefile = "tibia.pth"

    train_set = BoneDataset(dataset_df, bone_type)
    test_set = BoneDataset(dataset_df, bone_type, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = BoneModel()
    if os.path.exists(model_savefile):
        model.load_state_dict(torch.load(model_savefile, weights_only=True))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        correct_preds = 0
        total_preds = 0
        batch = 0
        for inputs, labels in train_loader:
            pictures = []
            for filename in inputs:
                image_array = transform_image(filename)
                pictures.append(image_array)
            
            optimizer.zero_grad()
            pictures = torch.tensor(np.array(pictures), dtype=torch.float32)
            labels = labels.to(torch.float32)
            outputs = model(pictures)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            batch_correct = torch.sum((torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).to(torch.float32)).item()
            correct_preds += batch_correct
            total_preds += len(labels)

            print(f"Training Batch = {batch + 1}, Batch Accuracy = {round(batch_correct / len(labels), 2)}, Batch Loss = {round(loss.item(), 4)}")
            batch += 1
        
            gc.collect()
        
        print(f"Epoch Training Accuracy = {round(correct_preds / total_preds, 2)}, Epoch Loss = {round(epoch_loss, 4)}")

        print("Saving Model")
        torch.save(model.state_dict(), model_savefile)
        print("Model Saved")

        model.eval()
        epoch_loss = 0
        correct_preds = 0
        total_preds = 0
        batch = 0
        for inputs, labels in test_loader:
            pictures = []
            for filename in inputs:
                image_array = transform_image(filename)
                pictures.append(image_array)
            
            pictures = torch.tensor(np.array(pictures), dtype=torch.float32)
            labels = labels.to(torch.float32)
            outputs = model(pictures)
            
            batch_correct = torch.sum((torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).to(torch.float32)).item()
            correct_preds += batch_correct
            total_preds += len(labels)

            print(f"Testing Batch = {batch + 1}, Batch Accuracy = {round(batch_correct / len(labels), 2)}")
            batch += 1
        
            gc.collect()
        
        print(f"Epoch Testing Accuracy = {round(correct_preds / total_preds, 2)}")
        model.train()

if __name__=="__main__":
    main()