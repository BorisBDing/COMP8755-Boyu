from model.clip_model import CLIPModel
from model.pretrained_models import _TRANSFORMER_MODELS
from transformers import AutoModel, AutoTokenizer
                
import torch

class TransformerModel(CLIPModel):
    def __init__(
        self,
        name: str,
        device: str = 'cpu',
        jit: bool = False,
        dtype: str = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._model_name = name
        model_path =_TRANSFORMER_MODELS[name]
        self._model = AutoModel.from_pretrained(str(model_path))

    @staticmethod
    def get_model_name(name: str):
        if '::' in name:
            model_name, pretrained = name.split('::')
        else:
            model_name = name
        return model_name.replace('/', '-')

    def encode_text(self, input_ids: 'torch.Tensor', **kwargs):
        return self._model.get_text_features(input_ids)

    def encode_image(self, pixel_values: 'torch.Tensor', **kwargs):
        return self._model.get_image_features(pixel_values)
    
    @property
    def image_size(self):
        return self._model.vision_model.embeddings.image_size
