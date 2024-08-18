import argparse
import os
from PIL import Image
import flwr as fl
import torch
from collections import OrderedDict
from flwr.client import NumPyClient, ClientApp
import warnings
from model import train,test,get_client_model,get_model_params 
from dataset import ClientDataset,ImagePathsDataset
from server import date_and_time
warnings.filterwarnings("ignore")
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datetime import datetime
import time 
start_time=time.time()
# current_datetime = datetime.now()
# formatted_datetime = current_datetime.strftime("%Y_%m_%d-%H_%M_%S")
parser = argparse.ArgumentParser(description="Flower")
"""parser.add_argument(
    "--dry",
    type=bool,
    default=False,
    required=False,
    help="Do a dry-run to check the client",
)"""
formatted_datetime=date_and_time()
parser.add_argument(
    "client_id",
    type=int,
    help="Client ID",
    default=False,
    
)
args = parser.parse_args()
client_id=args.client_id

os.makedirs(f"./models/run_{formatted_datetime}/client_models/clientid_{client_id}",exist_ok=True)




client_path=f"./celeba_500/celeba/custom_celeba_500/clients_train/client_{client_id}"
client_dataset=ClientDataset(client_path)
trainloader,valloader=client_dataset.load_client_data()
num_classes=client_dataset.get_num_classes()
print(f"Number of classes: {num_classes} exists in Client {client_id} data.")
def data_device(loader):
    for image,label in loader:
        image,label=image.to(DEVICE),label.to(DEVICE)
    return loader
#trainloader,valloader=data_device(trainloader),data_device(valloader)



net=get_client_model(client_id,num_classes)


#print("Device of the model is: " ,net.device)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_id,
        trainloader,
        valloader,
        device,
        net,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.client_id = client_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.validation_split = validation_split
        self.net=net
    #def get_parameters(self, config):
    #    print("Getting parameters for client ...")
    #    return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def get_parameters(self, config):
        print("Getting parameters for client ...")
        # Get all parameters from the model's state dictionary
        all_params = self.net.state_dict()
        # Filter out parameters related to the fully connected (fc) layer
        filtered_params = {name: param for name, param in all_params.items() if 'fc' not in name}
        # Convert parameters to NumPy arrays and return
        return [val.cpu().numpy() for _, val in filtered_params.items()]

    # def set_parameters(self, parameters):
    #     print("Setting parameters for client...")
    #     all_params = self.net.state_dict()
    #     filtered_params = {name: param for name, param in all_params.items() if 'fc' not in name}
        
    #     # Ensure that the number of parameters matches
    #     if len(filtered_params) != len(parameters):
    #         raise ValueError("Number of parameters provided does not match the number of parameters to set.")

    #     # Load parameters into the model
    #     state_dict = OrderedDict({name: torch.tensor(param) for name, param in zip(filtered_params.keys(), parameters)})
    #     self.net.load_state_dict(state_dict, strict=True)
    def set_parameters(self, parameters):
        print("Setting parameters for client...")
        all_params = self.net.state_dict()
        filtered_params = {name: param for name, param in all_params.items() if 'fc' not in name}
        
        # Ensure that the number of parameters matches
        if len(filtered_params) != len(parameters):
            raise ValueError("Number of parameters provided does not match the number of parameters to set.")

        # Load parameters into the model
        for name, param in zip(filtered_params.keys(), parameters):
            if name in self.net.state_dict():
                # Load the parameter if it exists in the model
                self.net.state_dict()[name].copy_(torch.tensor(param))
            else:
                print(f"Parameter {name} not found in model, skipping...")



    def fit(self, parameters, config):
        print("Training is starting for client ...")
        self.set_parameters(parameters)
        batch_size : int = config["batch_size"]
        epochs: int = config["local_epochs"]
        round_number: int = config["round_number"] 
        train(self.net, self.trainloader,self.valloader, epochs)

        torch.save(self.net.state_dict(), f"./models/run_{formatted_datetime}/client_models/clientid_{client_id}/model_round_{round_number}.pth")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Evaluation started for client ...")
        self.set_parameters(parameters)
        steps: int = config["val_steps"]
        round_number: int = config["round_number"]
        loss, accuracy = test(self.net, self.valloader)
        #torch.save(self.net.state_dict(), f"./models/client_models/{formatted_datetime}/{self.client_id}/model_round_{round_number}.pth")

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    def get_client_model(self):
        return self.net
    
"""def client_dry_run(device: torch.device = "cpu"):
    ""Weak tests to check whether all client methods are working as expected.""

    model = load_model(client_id=client_id)
    #trainloader,valloader=trainloader,valloader
  
    client = FlowerClient(trainloader, valloader,testloader,DEVICE,net)
    client.fit(
        client.get_parameters(config={"batch_size": 32, "local_epochs": 3})
    )

    client.evaluate(client.get_parameters(), {"val_steps": 32})

    print("Dry Run Successful")"""




def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient(client_id,trainloader,valloader,DEVICE,net).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


if __name__ == "__main__":
    from flwr.client import start_client


    print("Starting client...")
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id,trainloader,valloader,DEVICE,net).to_client(),
    )
