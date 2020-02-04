from abc import abstractmethod, ABC


class BaseLayer(ABC):

    @abstractmethod
    def forward(self, image):
        ...

    @abstractmethod
    def back(self, activation_theta, activation):
        ...

    @abstractmethod
    def update_weights(self, input_batch, error_batch, learning_rate):
        ...

    def save(self, folder: str):
        return ""
