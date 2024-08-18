import torch
from typing import List, Tuple
import flwr as fl
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar, EvaluateRes
#from client_barbara import testloader
from torchvision import transforms
#from client_anne import testloader
from torchvision.datasets import ImageFolder
from model import load_centralized_model,train,test,get_model_params,CentralizedNet
from dataset import load_centralized_test
import warnings
from datetime import datetime
import os
import os



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


warnings.filterwarnings("ignore")
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_folder_path(folder_name):
    train_path=f"client-veri/cropped/train/{folder_name}"
    test_path=f"client-veri/cropped/test/{folder_name}"
    return train_path,test_path


def data_device(loader):
    for image,label in loader:
        image.to(DEVICE),label.to(DEVICE)
    return loader

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d-%H_%M")
def date_and_time():
    return formatted_datetime
centralized_net=load_centralized_model()
centralized_model_param=get_model_params(centralized_net)
print(f"Centralized model output layer :{centralized_net.fc.out_features}")


testload=load_centralized_test(batch_size=32, shuffle=False)
testloader=data_device(testload)


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
        "round_number" : server_round
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps, "round_number" : server_round}


# # Define metric aggregation function
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
#strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
def get_evaluate_fn(net,testloader):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = net
        model.remove_fc_layer()
        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.set_fc_layer(model.num_classes)
        model.to(DEVICE)
        # call test
        loss, accuracy = test(
            net, testloader
        )  # <-------------------------- calls the `test` function, just what we did in the centralised setting
        return loss, {"accuracy": accuracy}

    return evaluate_fn

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights and update specific layers of the global model"""
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        print(server_round)
        #global_model = self._load_global_model()
        
        if server_round==1:
            global_model = self._load_global_model()

            self.update_global_model_parts(global_model,server_round, aggregated_parameters)
            self._save_global_model(global_model, server_round)
        elif server_round>1:
            global_model = self.load_last_global_model(server_round-1)
            self.update_global_model_parts(global_model,server_round, aggregated_parameters)
            self._save_global_model(global_model, server_round)  # Load global model
        
            # Update specific layers of the global model with client model's layers
            
        #     for client_proxy, fit_res in results:
        #         client_model = client_proxy.get_client_model()  # Get client model
        #         self.update_global_model(global_model, client_model)
            
        # try:    # Save the updated global model
        #     self._save_global_model(global_model, server_round)
        #     self.update_global_model_parts(global_model, aggregated_parameters)
        # except Exception as e:
        #     print(e)
        #     print("Aggregated parameters are None, global model is not updated")

        return aggregated_parameters, aggregated_metrics
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
    
    def load_last_global_model(self,server_round):
        model_path=f"models/run_{formatted_datetime}/global_models/model_round_{server_round}.pth"
        net=CentralizedNet().to(DEVICE)
        net.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
        return net
    def _load_global_model(self):
        """Load global model"""
        net = CentralizedNet().to(DEVICE)
        model_path = 'C:/Users/DELL/Desktop/total/mobilenetv2/Mobilenetv2/model.pth'
        net.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
        return net
    
    # def update_global_model_parts(self, global_model, aggregated_parameters):
    #     """Update specific parts of the global model with corresponding aggregated parameters"""
    #     # Get the aggregated parameters for the first three layers
    #     aggregated_params_first_three_layers = aggregated_parameters[:3]

    #     # Update the first three layers of the global model with the aggregated parameters
    #     for idx, (layer_name, params) in enumerate(aggregated_params_first_three_layers):
    #         # Update the parameters of the corresponding layer in the global model
    #         global_model.features[idx].load_state_dict({"weight": torch.tensor(params[0]), "bias": torch.tensor(params[1])})
    def update_global_model_parts(self, global_model,server_round, aggregated_parameters):
       if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(global_model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict if 'fc' not in k})
            global_model.remove_fc_layer()
            global_model.load_state_dict(state_dict, strict=True)
            global_model.set_fc_layer(global_model.num_classes)
            
    def _save_global_model(self, global_model, server_round):
        """Save updated global model"""
        print(f"Saving global model for {server_round}th round...")
        os.makedirs(f"models/run_{formatted_datetime}/global_models", exist_ok=True)
        
        torch.save(global_model.state_dict(), f"models/run_{formatted_datetime}/global_models/model_round_{server_round}.pth")
        

    # def update_global_model(self, global_model, client_model):
    #     """Update specific layers of the global model with client model's layers"""
    #     # Check if the layers exist in both models before loading parameters
    #     if hasattr(global_model, "features") and hasattr(client_model, "features"):
    #         global_model.features[0].load_state_dict(client_model.features[0].state_dict())
    #         global_model.features[1].load_state_dict(client_model.features[1].state_dict())
    #         global_model.features[2].load_state_dict(client_model.features[2].state_dict())
    #         global_model.avgpool.load_state_dict(client_model.avgpool.state_dict())
    #         global_model.dropout.load_state_dict(client_model.dropout.state_dict())
    #     else:
    #         raise ValueError("Global and client models have incompatible architectures")

            

  
    
strategy=CustomStrategy(
        fraction_fit=1,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=1,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(centralized_net, testloader),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        #initial_parameters=fl.common.ndarrays_to_parameters(centralized_model_param),
)


# Define config
config = ServerConfig(num_rounds=10)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

