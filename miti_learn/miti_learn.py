import timm
import numpy as np
import torch
from tqdm import tqdm


class MitiLearn:
    def __init__(self):
        """
        MitiLearn class is a generalized class for creating and training models for quantum error mitigation through machine learning
        """

        self.model = None
        self.input_data = None
        self.model_name = None
        self.model_type = None
        self.model_params = None
        self.isPretrained = True

    def load_model(self, model_name: str, isPretrained: bool = True):
        """
        Load model from timm library and check if it is in the list of available models
        If not, support custom models in the future.
        """
        self.model_name = model_name

        if model_name in timm.list_models():
            return timm.create_model(model_name, pretrained=isPretrained)
        else:
            # Import torch model
            return torch.load(model_name)

    def apply_model(self, model, input_data: np.ndarray):
        """
        Apply model to input data from quantum circuit
        """
        self.model = model
        self.input_data = input_data

        return self.model(self.input_data)

    def get_expectation_value(self, model_output: np.ndarray):
        """
        Get expectation value from model output
        """
        return np.mean(model_output)

    def train_model(
        self,
        model,
        input_data: np.ndarray,
        target: np.ndarray,
        optimizer,
        loss_fn,
        epochs: int,
    ):
        """
        Train model with input data and target
        """
        self.model = model
        self.input_data = input_data
        self.optimizer = optimizer | torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = loss_fn | torch.nn.MSELoss()
        self.epochs = epochs | 100

        # Detect if model is a torch model
        if hasattr(self.model, "train"):
            self.model.train()
            for epoch in tqdm(range(self.epochs)):
                self.optimizer.zero_grad()
                output = self.model(self.input_data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss {loss.item()}")

            return self.model

        else:
            raise ValueError("Model is not a torch model")

    def evaluate_model(self, model, input_data: np.ndarray, target: np.ndarray):
        """
        Evaluate model with input data and target
        """
        self.model = model
        self.input_data = input_data
        self.target = target

        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input_data)
            loss = self.loss_fn(output, target)

        return loss

    def save_model(self, model, path: str):
        """
        Save model to file
        """
        self.model = model
        self.model.save(path)

    def load_model_from_file(self, path: str):
        """
        Load model from file
        """

        return self.model.load(path)
