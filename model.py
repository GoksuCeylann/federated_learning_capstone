import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

import warnings
#from client import client_id,person_name,batch_size,get_folder_path
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the neural network


class CentralizedNet(nn.Module):
    def __init__(self,):
        super(CentralizedNet, self).__init__()
        # Load pretrained MobileNetV2 model
        pretrained_model = models.mobilenet_v2()
        #self.client_id=client_id
        self.num_classes=105
        # Extract the features part of the model
        self.features = pretrained_model.features
        
        # Define additional layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1280, self.num_classes)  # Adjusted based on MobileNetV2 architecture
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten features for the fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False
    
    def client_fc(self):
        self.fc=nn.Linear(1280, self.new_num_classes)
        return self.fc
    def set_fc_layer(self, num_classes):
        # Set the number of classes for the fully connected (fc) layer
        self.fc = nn.Linear(1280, num_classes)
        self.fc.to(DEVICE)
    def remove_fc_layer(self):
        # Remove the fully connected (fc) layer
        del self.fc
        
class ClientNet(nn.Module):
    def __init__(self,cid,new_num_classes):
        super(ClientNet, self).__init__()
        self.cid=cid
        # Load pretrained MobileNetV2 model
        pretrained_model = models.mobilenet_v2()
        #self.client_id=client_id
        self.new_num_classes=new_num_classes
        self.client_num_classes=self.new_num_classes+105
        self.num_classes=105
        # Extract the features part of the model
        self.features = pretrained_model.features
        
        # Define additional layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1280, self.client_num_classes)
        #self.fc_client=
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        # x = self.fc(x)
        # x = self.softmax(x)
        #output_client=self.fc_client(x)
        #output_combined = torch.cat((output_cent, output_client[:, -self.num_new_classes:]), dim=1)
        #output_combined = self.softmax(output_combined)
        return x
    def freeze(self):
        for param in self.features[3:17].parameters():
            param.requires_grad = False
    
    def client_fc(self):
        self.fc=nn.Linear(1280, self.new_num_classes)
        return self.fc

def train(net, trainloader,valloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    #print("Net device:" ,net.device)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    
"""   train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results"""


def test(net, testloader,steps: int=None,device: torch.device=DEVICE):
    """Evaluate the network on the entire test set."""
    print("Testing the network...")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def load_centralized_model():
    net=CentralizedNet().to(DEVICE)
    model_path = 'C:/Users/DELL/Desktop/total/mobilenetv2/Mobilenetv2/model.pth'
    net.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    print("Global Model loaded successfully")
    return net

def get_client_model(client_id,new_num_classes):
    net=ClientNet(client_id,new_num_classes)
    cent_net=load_centralized_model()
    #model_path = 'D:/CAPSTONE PROJECT/asıl/mobilenetv2/Mobilenetv2/model.pth'
    #state_dict = torch.load(model_path, map_location=torch.device('cuda'))
    state_dict=cent_net.state_dict()
    # Load weights until the last layer before the fully connected layer
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in state_dict and "fc" not in k}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    with torch.no_grad():
        net.fc.weight[:105].copy_(cent_net.fc.weight[:105])
        net.fc.bias[:105].copy_(cent_net.fc.bias[:105])
    #net.fc=net.client_fc()
    
    net.freeze()
    net.to(DEVICE)
    return net
    

def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

