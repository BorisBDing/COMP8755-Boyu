import torch
from typing import List, Union
from model.pretrained_models import (
    _TRANSFORMER_MODELS,
)


class Tokenizer:
    def __init__(self, name: str, **kwargs):
        self._name = name
        if name in _TRANSFORMER_MODELS:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(_TRANSFORMER_MODELS[name]))
        else:
            from model.simple_tokenizer import SimpleTokenizer

            self._tokenizer = SimpleTokenizer()

    def __call__(
        self,
        texts: Union[str, List[str]],
        context_length: int = 77,
        truncate: bool = True,
    ):
        """
        :param texts: An input string or a list of input strings to tokenize
        :param context_length: The context length to use; all English CLIP models use 77 as the context length.
            for Chinese CLIP models, context_length = 52, if the number of characters is bigger than 50, sentence will be truncate and omit the part left
        :param truncate: Whether to truncate the text in case its encoding is longer than the context length.

        :return: A dict of tokenized representations of the input strings and their corresponding attention masks with both
            shape = [batch size, context_length]
        """
        return self._tokenize(
                texts, context_length=context_length, truncate=truncate
            )

    def _tokenize(
        self,
        texts: Union[str, List[str]],
        context_length: int = 77,
        truncate: bool = True,
    ) -> dict:
        if isinstance(texts, str):
            texts = [texts]
        if self._name in _TRANSFORMER_MODELS:
            result = self._tokenizer(
                texts,
                max_length=context_length,
                return_attention_mask=True,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            return {
                'input_ids': result['input_ids'],
                'attention_mask': result['attention_mask'],
            }
        else:
            sot_token = self._tokenizer.encoder['<|startoftext|>']
            eot_token = self._tokenizer.encoder['<|endoftext|>']
            all_tokens = [
                [sot_token] + self._tokenizer.encode(text) + [eot_token]
                for text in texts
            ]

            input_ids = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
            attention_mask = torch.zeros(
                len(all_tokens), context_length, dtype=torch.long
            )

            for i, tokens in enumerate(all_tokens):
                if len(tokens) > context_length:
                    if truncate:
                        tokens = tokens[:context_length]
                        tokens[-1] = eot_token
                    else:
                        raise RuntimeError(
                            f'Input {texts[i]} is too long for context length {context_length}'
                        )
                input_ids[i, : len(tokens)] = torch.tensor(tokens)
                attention_mask[i, : len(tokens)] = 1

            return {'input_ids': input_ids, 'attention_mask': attention_mask}
