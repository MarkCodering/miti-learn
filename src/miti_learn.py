import timm
import numpy as np


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
            raise ValueError(f"Model {model_name} not found in timm list")

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
