from abc import abstractmethod, ABC


class BaseLayer(ABC):

    @abstractmethod
    def forward(self, image):
        ...

    @abstractmethod
    def back(self, activation_theta, activation):
        ...

    @abstractmethod
    def update_weights(self, layer_cache, learning_rate):
        ...

    def save(self, folder: str):
        return ""
