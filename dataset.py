import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader, Dataset 
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import os
import random
from glob import glob
import torch
#from client import client_id,person_name,batch_size,get_folder_path
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ImagePathsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths,class_indexes, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.class_indexes=class_indexes

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        class_name = os.path.basename(os.path.dirname(image_path))
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, int(self.class_indexes[class_name])  # class_name'i doğrudan döndür

    def __len__(self):
        return len(self.image_paths)


class ClientDataset():
    def __init__(self, folder_path, train_ratio=0.8):
        self.client_folder_paths = folder_path
        self.train_ratio = train_ratio
        self.class_names = sorted(os.listdir(folder_path))
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.class_indexes=self.get_class_names()
    def get_class_names(self):
        class_indexes = {}
        for idx, class_name in enumerate(self.class_names):
            class_indexes[class_name] = f"{idx+105}"
        return class_indexes

    def split_images(self, folder_path, class_names):
        train_list_full, val_list_full = [], []

        for class_name in class_names:
            class_folder = os.path.join(folder_path, class_name)
            image_paths = glob(class_folder+"/*")
            train_list, val_list = train_test_split(image_paths, train_size=int(self.train_ratio*len(image_paths)), random_state=42)
            train_list_full.extend(train_list)
            val_list_full.extend(val_list)
        return train_list_full, val_list_full
    
    def get_num_classes(self):
        return len(self.class_names)

    def load_client_data(self, batch_size=32):
        train_list, val_list = self.split_images(self.client_folder_paths, self.class_names)
        train_dataset = ImagePathsDataset(train_list,self.class_indexes, transform=self.transform)
        val_dataset = ImagePathsDataset(val_list, self.class_indexes,transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader



def load_centralized_test(batch_size=32, shuffle=True):
    # Define transformations for the images
    data_path = "./client-veri/cropped/test"  # Path to the folder containing celebrity name folders
    classes = ["Anne Hathaway", "Anthony Mackie", "Avril Lavigne","Ben Affleck" , "Bill Gates", "Barack Obama","Barbara Palvin" ]
    class_indices = [6, 91, 9, 8, 10, 7, 92]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Map class names to class indices
    class_to_idx = {classes[i]: class_indices[i] for i in range(len(classes))}

    # Create a dataset from the ImageFolder with custom class_to_idx mapping
    dataset = ImageFolder(root=data_path, transform=transform, target_transform=lambda x: class_to_idx[classes[x]])

    # Create a DataLoader with the custom dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader






"""if __name__ == "__main__":
    net=load_model()
    trainloader, valloader = load_data()
    train(net, trainloader, 5)
    loss, accuracy = test(net, valloader)
    print(f"Validation loss: {loss}, accuracy: {accuracy}")"""