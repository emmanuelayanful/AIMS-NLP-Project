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
        metadata = {
            "help": "Hugging Face API token for authentication."
        }
    )
    dataset_name: str = field(
        default="allenai/wmt22_african", 
        metadata = {
            "help": "Dataset to use."
        }
    )
    source_langs: List[str] = field(
        default = lambda: ["eng"], 
        metadata = {
            "help": "List of source languages."
        }
    )
    target_langs: List[str] = field(
        default = lambda: ["xho"],
        metadata={
            "help": "List of target languages."
        }
    )
    comet_model: str = field(
        default = "masakhane/africomet-qe-stl-1.1", 
        metadata = {
            "help": "COMET-QE model path."
        }
    )
    batch_size: int = field(
        default = 1024, 
        metadata = {
            "help": "Batch size for processing."
        }
    )
    top_k_train: int = field(
        default = 100, 
        metadata = {
            "help": "Number of top train sentences to retain."
        }
    )
    top_k_val: int = field(
        default = 100, 
        metadata={
            "help": "Number of top validation sentences to retain."
        }
    )
    top_k_test: int = field(
        default = 100, 
        metadata = {
            "help": "Number of top test sentences to retain."
        }
    )
    output_dir: str = field(
        default = "./output", 
        metadata = {
            "help": "Output directory for saved datasets."
        }
    )
    device: str = field(
        default = "cuda", 
        metadata={
            "help": "Device to run inference on (cuda or cpu)."
        }
    )
    max_size: Optional[int] = field(
        default = None, 
        metadata = {
            "help": "Maximum size of the train dev test sequence."
        }
    )
    seed: Optional[int] = field(
        default = 42, 
        metadata = {
            "help": "Random seed for shuffling dataset."
        }
    )

def get_dataset(dataset_name, language_pair):
    dataset = load_dataset(dataset_name, language_pair)
    temp = dataset["train"].train_test_split(test_size=0.3)
    temp2 = temp["test"].train_test_split(test_size=0.5)
    return DatasetDict(
        {
            "train": temp["train"], 
            "dev": temp2["train"], 
            "test": temp2["test"]
        }
    )

def format_to_comet(examples, direction):
    src, tgt = direction.split("-")
    return {
        "translation": [{"src": ex[src], "mt": ex[tgt]} for ex in examples["translation"]]
    }

def get_top_sentences(dataset, model, batch_size, k, device, seed):
    top_k_heap = []
    dataset = dataset.shuffle(seed=seed)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        translations = batch["translation"]
        scores = list(map(
            float, model.predict(translations, gpus=1 if device == "cuda" else 0)[0])
        )

        for translation, score in zip(translations, scores):
            heapq.heappush(top_k_heap, (score, translation["src"], translation["mt"]))
            if len(top_k_heap) > k:
                heapq.heappop(top_k_heap)

    return [
        {"translation": {"src": src, "mt": mt}} for _, src, mt in sorted(top_k_heap, key=lambda x: -x[0])
    ]

def convert_json(data, filename, direction):
    src, tgt = direction.split("-")
    
    def rename_keys(example):
        return {
            "translation": {
                src: example["translation"]["src"], 
                tgt: example["translation"]["mt"]
            }
        }

    mapped_dataset = data.map(rename_keys)
    dataset_list = mapped_dataset.to_pandas().to_dict(orient="records")
    
    with jsonlines.open(filename, "w") as writer:
        writer.write_all(dataset_list)

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    if script_args.max_size is None:
        train_max_size = script_args.top_k_train * 2
        val_max_size = script_args.top_k_val * 2
        test_max_size = script_args.top_k_test * 2
    else:
        train_max_size = script_args.max_size
        val_max_size = script_args.max_size
        test_max_size = script_args.max_size

    logger.info("Logging into huggingface.")
    login(token=script_args.hf_token)

    logger.info("Downloading COMET-QE model.")
    model_path = download_model(script_args.comet_model)
    model = load_from_checkpoint(model_path)
    
    for src_lang in script_args.source_langs:
        for tgt_lang in script_args.target_langs:
            if src_lang == tgt_lang:
                continue
            
            language_pair = f"{src_lang}-{tgt_lang}"
            try:
                logger.info(f"\nProcessing language pair: {language_pair}")
                raw_dataset = get_dataset(script_args.dataset_name, language_pair)
            except:
                logger.error(f"Failed to find dataset for language pair {language_pair}.")
                logger.info("Skipping to next language pair...")
                continue
            
            logger.info("\nConverting dataset COMET-QE format.")
            
            train = raw_dataset["train"].select_columns(
                ["translation"]
            ).shuffle(seed=script_args.seed).select(
                range(train_max_size)
            ).map(
                lambda x: format_to_comet(x, language_pair), 
                batched=True
            )
            val = raw_dataset["dev"].select_columns(
                ["translation"]
            ).shuffle(seed=script_args.seed).select(
                range(val_max_size)
            ).map(
                lambda x: format_to_comet(x, language_pair), 
                batched=True
            )
            test = raw_dataset["test"].select_columns(
                ["translation"]
            ).shuffle(seed=script_args.seed).select(
                range(test_max_size)
            ).map(
                lambda x: format_to_comet(x, language_pair), 
                batched=True
            )

            logger.info("\nSelecting top-scoring sentences.\n")
            train_sentences = get_top_sentences(
                train, model, script_args.batch_size, 
                script_args.top_k_train, script_args.device, 
                script_args.seed
            )
            val_sentences = get_top_sentences(
                val, model, script_args.batch_size, 
                script_args.top_k_val, script_args.device, 
                script_args.seed
            )
            test_sentences = get_top_sentences(
                test, model, script_args.batch_size, 
                script_args.top_k_test, script_args.device, 
                script_args.seed
            )
            
            logger.info("\nConverting dataset to JSON format.")
            comet_model = script_args.comet_model.split("/")[-1]
            os.makedirs(
                os.path.join(script_args.output_dir, comet_model, language_pair), 
                exist_ok=True
            )
            convert_json(
                Dataset.from_list(train_sentences), 
                os.path.join(
                    script_args.output_dir, 
                    comet_model,
                    language_pair, 
                    "train.json"
                ), 
                language_pair
            )
            convert_json(
                Dataset.from_list(val_sentences), 
                os.path.join(
                    script_args.output_dir, 
                    comet_model,
                    language_pair, 
                    "dev.json"
                ), 
                language_pair
            )
            convert_json(
                Dataset.from_list(test_sentences), 
                os.path.join(
                    script_args.output_dir, 
                    comet_model,
                    language_pair, 
                    "test.json"
                ), 
                language_pair
            )
    
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
