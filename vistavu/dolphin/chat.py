"""
Dolphin Chat Module

This module defines the DOLPHIN class, which wraps the DonutModel and SwinEncoder
to provide image-based chat and inference capabilities.
"""

import os
import warnings
from collections import OrderedDict
from typing import List, Union, Optional, Tuple, Any, Dict

import torch
from omegaconf import ListConfig, DictConfig
from PIL import Image
from transformers import PreTrainedTokenizerFast

# Updated imports for new package structure
from .utils.model import DonutConfig, DonutModel, SwinEncoder
from .utils.processor import DolphinProcessor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def try_rename_legacy_weights(ckpt: Dict[str, Any], output_path: str = "") -> OrderedDict:
    """
    Rename legacy checkpoints to match the current model architecture.

    Args:
        ckpt (Dict[str, Any]): The checkpoint dictionary.
        output_path (str, optional): Path to save the renamed checkpoint. Defaults to "".

    Returns:
        OrderedDict: The renamed checkpoint state dictionary.
    """
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    if "module" in ckpt.keys():
        ckpt = ckpt["module"]
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("encoder"):
            new_ckpt["vpm" + k[len("encoder") :]] = v
        elif k.startswith("decoder"):
            new_ckpt["llm" + k[len("encoder") :]] = v
        else:
            new_ckpt[k] = v
    if output_path:
        torch.save(new_ckpt, output_path)
    return new_ckpt


def convert_listconfig_to_list(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OmegaConf ListConfig objects to standard Python lists.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        Dict[str, Any]: The configuration with ListConfigs converted to lists.
    """
    new_config = {}
    for k, v in config.items():
        if isinstance(v, ListConfig):
            new_config[k] = list(v)
        else:
            new_config[k] = v
    return new_config


class DOLPHIN:
    """
    Main class for the Dolphin VLM model interaction.
    """

    def __init__(self, config: DictConfig, ckpt_path: str = "") -> None:
        """
        Initialize the DOLPHIN model.

        Args:
            config (DictConfig): The model configuration.
            ckpt_path (str, optional): Path to a checkpoint. Defaults to "".
        """
        self.model_args = config.model
        self.swin_args = config.model.pop("swin_args")
        self.swin_args = convert_listconfig_to_list(self.swin_args)

        vision_tower = SwinEncoder(
            input_size=self.swin_args["img_size"],
            patch_size=self.swin_args["patch_size"],
            embed_dim=self.swin_args["embed_dim"],
            window_size=self.swin_args["window_size"],
            encoder_layer=self.swin_args["encoder_layer"],
            num_heads=self.swin_args["num_heads"],
            align_long_axis=self.swin_args["align_long_axis"],
        )

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.model_args.tokenizer_path)
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.unk_token = "<unk>"

        if self.model_args.get("extra_answer_tokens", False):
            # print("Allowing multitask training: adding <Answer/> to the tokenizer.")
            prompt_end_token = " <Answer/>"
            self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set([prompt_end_token]))})
            self.tokenizer._prompt_end_token = prompt_end_token
            self.tokenizer._prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(prompt_end_token)

        donut_config = DonutConfig(
            decoder_layer=self.model_args.decoder_layer,
            max_length=self.model_args.max_length,
            max_position_embeddings=self.model_args.max_position_embeddings,
            hidden_dimension=self.model_args.hidden_dimension,
        )

        self.model = DonutModel(config=donut_config, vision_tower=vision_tower, tokenizer=self.tokenizer)
        if self.model_args.model_name_or_path:
            ckpt = torch.load(self.model_args.model_name_or_path)
            # Fix typo in original code: try_rename_lagacy_weights -> try_rename_legacy_weights
            ckpt = try_rename_legacy_weights(ckpt)
            self.model.load_state_dict(ckpt, strict=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        transform_args = {
            "input_size": self.swin_args["img_size"],
            "max_length": self.model_args.max_length,
        }
        self.processor = DolphinProcessor({}, self.tokenizer, transform_args=transform_args)

    def chat(
        self,
        question: Union[str, List[str]],
        image: Union[str, Image.Image, List[Union[str, Image.Image]]],
        return_raw: bool = False,
        return_score: bool = False,
        return_img_size: bool = False,
        only_return_img_size: bool = False,
        max_batch_size: int = 16,
    ) -> Union[str, List[str], Tuple[Any, ...], Dict[str, Any], Any]:
        """
        Run inference (chat) with the model.

        Args:
            question (Union[str, List[str]]): The question or list of questions.
            image (Union[str, Image.Image, List[Union[str, Image.Image]]]): The image path, PIL Image, or list of them.
            return_raw (bool, optional): Whether to return raw model output. Defaults to False.
            return_score (bool, optional): Whether to return confidence scores. Defaults to False.
            return_img_size (bool, optional): Whether to return original image size. Defaults to False.
            only_return_img_size (bool, optional): Whether to only return image size and skip inference. Defaults to False.
            max_batch_size (int, optional): Maximum batch size for inference. Defaults to 16.

        Returns:
            Union[str, List[str], Tuple[Any, ...], Dict[str, Any], Any]: The model's response.
        """

        def _preprocess_image(image_input):
            if isinstance(image_input, str):
                image_input = Image.open(image_input).convert("RGB")
            if return_img_size or only_return_img_size:
                image_tensor, ori_size = self.processor.process_image_for_inference(image_input, return_img_size=True)
            else:
                image_tensor = self.processor.process_image_for_inference(image_input, return_img_size=False)
                ori_size = None
            return image_tensor, ori_size

        def _preprocess_prompt(question_input):
            if self.model_args.get("extra_answer_tokens", False):
                if self.tokenizer._prompt_end_token not in question_input:
                    question_input = question_input + self.tokenizer._prompt_end_token
            prompt_ids_out = self.processor.process_prompt_for_inference(question_input)
            return prompt_ids_out

        def _preprocess_prompt_batch(question_input):
            if self.model_args.get("extra_answer_tokens", False):
                for i in range(len(question_input)):
                    if self.tokenizer._prompt_end_token not in question_input[i]:
                        question_input[i] = question_input[i] + self.tokenizer._prompt_end_token
                    if not question_input[i].startswith("<s>"):
                        question_input[i] = "<s>" + question_input[i]
            return question_input

        def _postprocess(output_text, question_text):
            output_text = output_text.replace("<s>", "").replace(question_text, "").replace("</s>", "").replace("<pad>", "")
            if self.model_args.get("extra_answer_tokens", False):
                output_text = output_text.split(self.tokenizer._prompt_end_token)[-1]
            return output_text

        if isinstance(question, list):
            image_tensor_list = []
            # Ensure image is also a list if question is a list
            if not isinstance(image, list):
                # Handle edge case where one image is provided for multiple questions
                # (though the original code implies strict 1:1 mapping by iteration)
                pass # Original code assumes list 
            
            for i in image:
                image_tensor_item, _ = _preprocess_image(i)
                image_tensor_list.append(image_tensor_item)
            image_tensor = torch.cat(image_tensor_list, dim=0)

            question = _preprocess_prompt_batch(question)
            self.processor.tokenizer.padding_side = "left"
            prompt_ids = self.processor.tokenizer(
                question, add_special_tokens=False, return_tensors="pt", padding=True
            ).input_ids
            ori_size = None # Original logic doesn't seemingly collect all sizes for batch
        else:
            image_tensor, ori_size = _preprocess_image(image)
            prompt_ids = _preprocess_prompt(question)

        if only_return_img_size:
            return ori_size

        model_output_batch = []
        for i in range(0, image_tensor.shape[0], max_batch_size):
            image_tensor_batch = image_tensor[i : i + max_batch_size]
            prompt_ids_batch = prompt_ids[i : i + max_batch_size]
            model_output = self.model.inference(image_tensors=image_tensor_batch, prompt_ids=prompt_ids_batch)
            model_output_batch.append(model_output)
        
        model_output = {}
        # Aggregate batch results
        for k, v in model_output_batch[0].items():
            if isinstance(v, torch.Tensor):
                model_output[k] = sum(
                    [v_batch[k].cpu().numpy().tolist() for v_batch in model_output_batch],
                    [],
                )
            else:
                model_output[k] = sum([v_batch[k] for v_batch in model_output_batch], [])

        if return_raw:
            if return_img_size:
                return model_output, ori_size
            return model_output
        else:
            if isinstance(question, list):
                output = [_postprocess(model_output["repetitions"][i], question[i]) for i in range(len(question))]
                score = model_output["scores"]
            else:
                output = _postprocess(model_output["repetitions"][0], question)
                score = model_output["scores"][0]
            
            if return_score:
                return output, score
            if return_img_size:
                return output, ori_size
            return output
