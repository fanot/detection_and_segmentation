"""
Abstract class for all models
that consist of method that need to be implemented to model work well
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, data: str, imgsz: int, epochs: int, batch: int):
        """
        abstract method for train function
        """

        pass

    @abstractmethod
    def val(self, data: str, imgsz: int):
        """
        abstract method for evaluate function
        """

        pass

    @abstractmethod
    def predict(self, source: str):
        """
        abstract method for predict function
        """

        pass
