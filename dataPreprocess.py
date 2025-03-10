#!/usr/bin/env python
# coding=utf-8
"""
Script for selecting top-scoring machine translations using COMET-QE.
"""

import os
import logging
import sys
import heapq
import numpy as np
import jsonlines
from dataclasses import dataclass, field
from typing import Optional, List
import language_map as lm

from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from comet import download_model, load_from_checkpoint

from transformers import HfArgumentParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """
    Arguments for dataset and model configurations.
    """
    hf_token: str = field(
        metadata={"help": "Hugging Face API token for authentication."}
    )
    dataset_name: str = field(
        default="allenai/wmt22_african", 
        metadata={"help": "Dataset to use."}
    )
    source_langs: List[str] = field(
        default_factory=lambda: ["eng"], 
        metadata={"help": "List of source languages (language codes)."}
    )
    target_langs: List[str] = field(
        default_factory=lambda: ["xho"],
        metadata={"help": "List of target languages (language codes)."}
    )
    source_langs_names: List[str] = field(
        default_factory=lambda: ["English"], 
        metadata={"help": "List of source languages (full names)."}
    )
    target_langs_names: List[str] = field(
        default_factory=lambda: ["Xhosa"],
        metadata={"help": "List of target languages (full names)."}
    )
    comet_model: str = field(
        default="masakhane/africomet-qe-stl-1.1", 
        metadata={"help": "COMET-QE model path."}
    )
    batch_size: int = field(
        default=1024, 
        metadata={"help": "Batch size for processing."}
    )
    top_k_train: int = field(
        default=100, 
        metadata={"help": "Number of top train sentences to retain."}
    )
    top_k_val: int = field(
        default=100, 
        metadata={"help": "Number of top validation sentences to retain."}
    )
    top_k_test: int = field(
        default=100, 
        metadata={"help": "Number of top test sentences to retain."}
    )
    output_dir: str = field(
        default="./output", 
        metadata={"help": "Output directory for saved datasets."}
    )
    device: str = field(
        default="cuda", 
        metadata={"help": "Device to run inference on (cuda or cpu)."}
    )
    max_size: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum number of examples to select."}
    )
    seed: Optional[int] = field(
        default=42, 
        metadata={"help": "Random seed for shuffling dataset."}
    )

def get_dataset(dataset_name, language_pair):
    dataset = load_dataset(dataset_name, language_pair, trust_remote_code=True)
    if dataset_name == "allenai/wmt22_african":
        temp = dataset["train"].train_test_split(test_size=0.3)
        temp2 = temp["test"].train_test_split(test_size=0.5)
        return DatasetDict(
            {"train": temp["train"], "dev": temp2["train"], "test": temp2["test"]}
        )
    return dataset

def format_to_comet(examples, direction):
    src, tgt = direction.split("-")
    return {
        "translation": [{"src": ex[src], "mt": ex[tgt]} for ex in examples["translation"]]
    }

def convert_json(data, dataset_name, filename, direction, language_pair):
    src, tgt = direction.split("-")
    
    def rename_comet(example):
        return {
            "translation": {
                src: example["translation"]["src"], 
                tgt: example["translation"]["mt"]
            }
        }

    def rename_mafand(example):
        return {
            "translation": {
                src: example["translation"][language_pair.split("-")[0]], 
                tgt: example["translation"][language_pair.split("-")[1]]
            }
        }

    def rename_flores(example):
        src_ = language_pair.split("-")[0]
        tgt_ = language_pair.split("-")[1]
        return {
            "translation": {
                src: example[f"sentence_{src_}"], 
                tgt: example[f"sentence_{tgt_}"]
            }
        }

    if dataset_name == "allenai/wmt22_african":
        mapped_dataset = data.map(rename_comet)
    elif dataset_name == "masakhane/mafand":
        mapped_dataset = data.map(rename_mafand)
    else:
        mapped_dataset = data.map(rename_flores).select_columns(["translation"])
    dataset_list = mapped_dataset.to_pandas().to_dict(orient="records")
    
    with jsonlines.open(filename, "w") as writer:
        writer.write_all(dataset_list)

def merge_jsonlines(files, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        for file in files:
            with jsonlines.open(file, mode='r') as reader:
                for obj in reader:
                    writer.write(obj)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    logger.info("Logging into huggingface.")
    login(token=script_args.hf_token)
    
    for src_lang_id, src_lang in enumerate(script_args.source_langs):
        for tgt_lang_id, tgt_lang in enumerate(script_args.target_langs):
            if src_lang == tgt_lang:
                continue
            
            language_pair = f"{src_lang}-{tgt_lang}"
            
            logger.info(f"Processing language pair: {language_pair}")
            raw_dataset = get_dataset(script_args.dataset_name, language_pair)
            
            new_src_lang = lm.get_language_code(script_args.source_langs_names[src_lang_id])
            new_tgt_lang = lm.get_language_code(script_args.target_langs_names[tgt_lang_id])
            new_language_pair = f"{new_src_lang}-{new_tgt_lang}"

            dataset_name = script_args.dataset_name.split("/")[1]
            output_path = os.path.join(script_args.output_dir, dataset_name, new_language_pair)
            os.makedirs(output_path, exist_ok=True)
            
            if script_args.dataset_name == "allenai/wmt22_african":
                model_path = download_model(script_args.comet_model)
                model = load_from_checkpoint(model_path)
                
                train = raw_dataset["train"].shuffle(seed=script_args.seed).select(range(script_args.max_size)).map(lambda x: format_to_comet(x, language_pair), batched=True)
                val = raw_dataset["dev"].shuffle(seed=script_args.seed).select(range(script_args.max_size)).map(lambda x: format_to_comet(x, language_pair), batched=True)
                test = raw_dataset["test"].shuffle(seed=script_args.seed).select(range(script_args.max_size)).map(lambda x: format_to_comet(x, language_pair), batched=True)
                
                convert_json(train, script_args.dataset_name, os.path.join(output_path, "train.json"), new_language_pair, language_pair)
                convert_json(val, script_args.dataset_name, os.path.join(output_path, "dev.json"), new_language_pair, language_pair)
                convert_json(test, script_args.dataset_name, os.path.join(output_path, "test.json"), new_language_pair, language_pair)
            
            
            elif script_args.dataset_name in ["facebook/flores", "masakhane/mafand"]:
                for split in raw_dataset.keys():
                    convert_json(raw_dataset[split], script_args.dataset_name, os.path.join(output_path, f"{split}.json"), new_language_pair, language_pair)
                if script_args.dataset_name == "masakhane/mafand":
                    input_files = os.listdir(output_path)
                    input_files = [os.path.join(output_path, file) for file in input_files]
                    merge_jsonlines(input_files, os.path.join(output_path, "merged.json"))
    
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    main()