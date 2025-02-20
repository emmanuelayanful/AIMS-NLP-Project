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
from typing import Optional

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
    hf_token: str = field(metadata={"help": "Hugging Face API token for authentication."})
    dataset_name: str = field(default="allenai/wmt22_african", metadata={"help": "Dataset to use."})
    language_pair: str = field(default="eng-xho", metadata={"help": "Translation direction."})
    comet_model: str = field(default="masakhane/africomet-qe-stl-1.1", metadata={"help": "COMET-QE model path."})
    batch_size: int = field(default=1024, metadata={"help": "Batch size for processing."})
    top_k_train: int = field(default=100, metadata={"help": "Number of top train sentences to retain."})
    top_k_val: int = field(default=100, metadata={"help": "Number of top validation sentences to retain."})
    top_k_test: int = field(default=100, metadata={"help": "Number of top test sentences to retain."})
    output_dir: str = field(default="./output", metadata={"help": "Output directory for saved datasets."})
    device: str = field(default="cuda", metadata={"help": "Device to run inference on (cuda or cpu)."})
    max_size: Optional[int] = field(default=None, metadata={"help": "Maximum size of the train dev test sequence."})
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed for shuffling dataset."})


def get_dataset(dataset_name, language_pair):
    dataset = load_dataset(dataset_name, language_pair)
    temp = dataset["train"].train_test_split(test_size=0.3)
    temp2 = temp["test"].train_test_split(test_size=0.5)
    return DatasetDict({"train": temp["train"], "dev": temp2["train"], "test": temp2["test"]})


def format_to_comet(examples, direction):
    src, tgt = direction.split("-")
    return {"translation": [{"src": ex[src], "mt": ex[tgt]} for ex in examples["translation"]]}


def get_top_sentences(dataset, model, batch_size, k, device, seed):
    top_k_heap = []
    dataset = dataset.shuffle(seed=seed)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        translations = batch["translation"]
        scores = list(map(float, model.predict(translations, gpus=1 if device == "cuda" else 0)[0]))

        for translation, score in zip(translations, scores):
            heapq.heappush(top_k_heap, (score, translation["src"], translation["mt"]))
            if len(top_k_heap) > k:
                heapq.heappop(top_k_heap)

    return [{"translation": {"src": src, "mt": mt}} for _, src, mt in sorted(top_k_heap, key=lambda x: -x[0])]


def convert_json(data, filename, direction):
    src, tgt = direction.split("-")
    
    def rename_keys(example):
        return {"translation": {src: example["translation"]["src"], tgt: example["translation"]["mt"]}}

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
    
    logger.info("\nLogging into hugginface.")
    login(token=script_args.hf_token)
    logger.info("Logged in successfully.")
    
    logger.info("\nDownloading dataset and model.")
    raw_dataset = get_dataset(script_args.dataset_name, script_args.language_pair)
    
    model_path = download_model(script_args.comet_model)
    model = load_from_checkpoint(model_path)
    
    logger.info("\nSpliting dataset into tran dev test.")
    train, val, test = raw_dataset["train"], raw_dataset["dev"], raw_dataset["test"]
    
    logger.info("\nProcessing dataset to comet format.")
    
    train_comet = train.select_columns(["translation"]).shuffle(seed=42).select(
        range(train_max_size)
    ).map(
        lambda x: format_to_comet(x, script_args.language_pair), batched=True
    )
    val_comet = val.select_columns(["translation"]).shuffle(seed=42).select(
        range(val_max_size)
    ).map(
        lambda x: format_to_comet(x, script_args.language_pair), batched=True
    )
    test_comet = test.select_columns(["translation"]).shuffle(seed=42).select(
        range(test_max_size)
    ).map(
        lambda x: format_to_comet(x, script_args.language_pair), batched=True
    )

    logger.info("\nSelecting top sentences with the chosen comet model.")
    train_sentences = get_top_sentences(
        train_comet, model, script_args.batch_size, script_args.top_k_train, script_args.device, script_args.seed
    )
    val_sentences = get_top_sentences(
        val_comet, model, script_args.batch_size, script_args.top_k_val, script_args.device, script_args.seed
    )
    test_sentences = get_top_sentences(
        test_comet, model, script_args.batch_size, script_args.top_k_test, script_args.device, script_args.seed
    )
    
    logger.info("\nSaving datasets as jsonlines.")
    os.makedirs(script_args.output_dir, exist_ok=True)
    path = os.path.join(script_args.output_dir, script_args.language_pair)
    os.makedirs(path, exist_ok=True)
    convert_json(Dataset.from_list(train_sentences), os.path.join(path, "train.json"), script_args.language_pair)
    convert_json(Dataset.from_list(val_sentences), os.path.join(path, "dev.json"), script_args.language_pair)
    convert_json(Dataset.from_list(test_sentences), os.path.join(path, "test.json"), script_args.language_pair)
    
    logger.info("\nProcessing completed successfully.")


if __name__ == "__main__":
    main()
