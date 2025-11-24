from abc import ABC, abstractmethod


class DiffusionModelForCheckpointing(ABC):
    @abstractmethod
    def inference_preparation_phase(self, *args, **kwargs):
        pass

    @abstractmethod
    def inference_denoising_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def inference_aft_denoising(self, *args, **kwargs):
        pass
