import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence


import transformers
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from tinyllava.arguments import *
from tinyllava.utils import *
from tinyllava.data.process import *
from tinyllava.constants import *

###################
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

from PIL import Image
import torch.nn as nn
import math

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
##########################

ImageFile.LOAD_TRUNCATED_IMAGES = True

# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  data_args: DataArguments):
#         super(LazySupervisedDataset, self).__init__()
#         list_data_dict = json.load(open(data_path, "r"))

#         rank0_print("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         self.list_data_dict = list_data_dict
#         self.data_args = data_args

#     def __len__(self):
#         return len(self.list_data_dict)

#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             img_tokens = 128 if 'image' in sample else 0
#             length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
#         return length_list

#     @property
#     def modality_lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#             cur_len = cur_len if 'image' in sample else -cur_len
#             length_list.append(cur_len)
#         return length_list

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
#         if 'image' in sources[0]:
#             image_file = self.list_data_dict[i]['image']
#             image_folder = self.data_args.image_folder
#             processor = self.data_args.image_processor
#             image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
#             if self.data_args.image_aspect_ratio == 'pad':
#                 def expand2square(pil_img, background_color):
#                     width, height = pil_img.size
#                     if width == height:
#                         return pil_img
#                     elif width > height:
#                         result = Image.new(pil_img.mode, (width, width), background_color)
#                         result.paste(pil_img, (0, (width - height) // 2))
#                         return result
#                     else:
#                         result = Image.new(pil_img.mode, (height, height), background_color)
#                         result.paste(pil_img, ((height - width) // 2, 0))
#                         return result

#                 image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             else:
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]),
#                 self.data_args)
#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])
#         data_dict = preprocess(
#             sources,
#             self.tokenizer,
#             has_image=('image' in self.list_data_dict[i]))
#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         # image exist in the data
#         if 'image' in self.list_data_dict[i]:
#             data_dict['image'] = image
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
#         return data_dict


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances]
#                                   for key in ("input_ids", "labels"))
#         if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
#             for input_id in input_ids:
#                 input_id[input_id == self.tokenizer.eos_token_id] = -300
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(labels,
#                                                  batch_first=True,
#                                                  padding_value=IGNORE_INDEX)
#         input_ids = input_ids[:, :self.tokenizer.model_max_length]
#         attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
#         labels = labels[:, :self.tokenizer.model_max_length]
#         # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
#         # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
#         # FIXME: eos id first, and convert them back.
#         if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
#             for input_id in input_ids:
#                 input_id[input_id == -300] = self.tokenizer.eos_token_id

#         batch = dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=attention_mask,
#         )

#         if 'image' in instances[0]:
#             images = [instance['image'] for instance in instances]
#             if all(x is not None and x.shape == images[0].shape for x in images):
#                 batch['images'] = torch.stack(images)
#             else:
#                 batch['images'] = images

#         return batch


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
#                                 data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
#                                           data_path=data_args.data_path,
#                                           data_args=data_args)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset,
#                 eval_dataset=None,
#                 data_collator=data_collator)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.multimodal_cfg['image_folder']
            processor = self.multimodal_cfg['image_processor']
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except Exception as exn:
                print(exn)
                import random
                return random.choice(self)

            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
            elif self.multimodal_cfg['image_aspect_ratio'] == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # import pdb; pdb.set_trace()
            image_token_len = self.multimodal_cfg["image_token_len"]
            patch_size = int(image.shape[1]//math.sqrt(image_token_len))
            cur_token_len = (image.shape[1]//patch_size) * (image.shape[2]//patch_size)   # FIXME: 14 is hardcoded patch size

            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversatons"] for e in sources])

            sources = preprocess_multimodal(
                sources,
                self.multimodal_cfg, cur_token_len)
        else:
            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversatons"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.multimodal_cfg['is_multimodal']:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    image_token_len=data_args.image_token_len,
                                    image_folder=data_args.image_folder,
                                    image_aspect_ratio=data_args.image_aspect_ratio,
                                    use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
                                    image_processor=getattr(data_args, 'image_processor', None)))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
