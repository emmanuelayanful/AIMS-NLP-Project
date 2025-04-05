#!/usr/bin/env python
# coding=utf-8
"""
Script for selecting top-scoring machine translations using COMET-QE,
with added language identification (LID) filtering using FastText OpenLID.
If not enough valid examples are found after filtering, the script accumulates
additional examples until at least top_k_train samples are obtained (or the dataset is exhausted).
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
from datasets import Dataset

from huggingface_hub import login, hf_hub_download
from datasets import load_dataset, Dataset, DatasetDict
from comet import download_model, load_from_checkpoint

from transformers import HfArgumentParser

# Import FastText and load the OpenLID model from Hugging Face Hub
import fasttext
openlid_model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
openlid_model = fasttext.load_model(openlid_model_path)

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
    source_lang: str = field(
        default_factory=lambda: "eng", 
        metadata={"help": "Source language (language code)."}
    )
    target_lang: str = field(
        default_factory=lambda: "xho",
        metadata={"help": "Target language (language code)."}
    )
    source_lang_name: str = field(
        default_factory=lambda: "English", 
        metadata={"help": "Source language (full name)."}
    )
    target_lang_name: str = field(
        default_factory=lambda: "Xhosa",
        metadata={"help": "Target language (full name)."}
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

def is_valid_language(example, key, expected_lang):
    """
    Uses the FastText OpenLID model to check if the text under example["translation"][key]
    is in the expected language.
    Returns True if the detected language matches expected_lang, otherwise False.
    """
    try:
        source_text = example["translation"][key]
        labels, scores = openlid_model.predict(source_text)
        # The model returns labels in the form '__label__<code>'
        detected_lang = labels[0].replace("__label__", "")
        return detected_lang == expected_lang
    except Exception as e:
        return False

def accumulate_valid_examples(dataset, filter_func, desired_count):
    """
    Iterates over the dataset and accumulates examples for which filter_func(example)
    returns True. Stops once at least desired_count valid examples are found or
    when the dataset is exhausted.
    """
    valid_examples = []
    for example in dataset:
        if filter_func(example):
            valid_examples.append(example)
            if len(valid_examples) >= desired_count:
                break
    return valid_examples

def get_dataset(dataset_name, language_pair):
    dataset = load_dataset(dataset_name, language_pair, trust_remote_code=True)
    logger.info(f"Dataset for the {language_pair} loaded!")
    return dataset

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
        scores = list(map(float, model.predict(translations, gpus=1 if device == "cuda" else 0)[0]))

        for translation, score in zip(translations, scores):
            heapq.heappush(top_k_heap, (score, translation["src"], translation["mt"]))
            if len(top_k_heap) > k:
                heapq.heappop(top_k_heap)

    return Dataset.from_list([{"translation": {"src": src, "mt": mt}} for _, src, mt in sorted(top_k_heap, key=lambda x: -x[0])])

def convert_json(data, dataset_name, filename, direction, language_pair, comet_name):
    if "ewe" in language_pair and dataset_name == "allenai/nllb":
        tgt, src = direction.split("-")
    else:
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

    if dataset_name == "allenai/nllb":
        mapped_dataset = data.map(rename_comet).select_columns(["translation"]) if comet_name == "random" else data.map(rename_comet)
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
    
    # Determine maximum selection size if not provided
    if script_args.max_size is None:
        train_max_size = script_args.top_k_train * 2
        val_max_size = script_args.top_k_val * 2
        test_max_size = script_args.top_k_test * 2
    else:
        train_max_size = script_args.max_size
        val_max_size = script_args.max_size
        test_max_size = script_args.max_size
    
    logger.info("Logging into Hugging Face.")
    login(token=script_args.hf_token)
    src_lang, tgt_lang = script_args.source_lang, script_args.target_lang
    source_lang_name, target_lang_name = script_args.source_lang_name, script_args.target_lang_name

    language_pair = f"{src_lang}-{tgt_lang}"
    
    logger.info(f"Processing language pair: {language_pair}")
    raw_dataset = get_dataset(script_args.dataset_name, language_pair)
    
    # If "ewe" appears in src_lang, swap roles (as per your logic)
    if "ewe" in src_lang.lower():
        src_lang, tgt_lang, source_lang_name, target_lang_name = tgt_lang, src_lang, target_lang_name, source_lang_name
    
    new_src_lang = lm.get_language_code(source_lang_name)
    new_tgt_lang = lm.get_language_code(target_lang_name)
    new_language_pair = f"{new_src_lang}-{new_tgt_lang}"
    dataset_name = script_args.dataset_name.split("/")[1]
    
    # Decide which key to use when filtering based on your condition:
    # For instance, if the source language is "Ewe", use src_lang; otherwise use tgt_lang.
    # (Adjust this logic as needed.)
    key = tgt_lang
    expected_lang_code = key  # Based on your ternary logic
    
    # ---- Branch for COMET-QE model provided ----
    if script_args.dataset_name == "allenai/nllb" and script_args.comet_model != "None":
        # Download and load the COMET-QE model
        model_path = download_model(script_args.comet_model)
        model = load_from_checkpoint(model_path)

        # Shuffle the training dataset
        shuffled_train = raw_dataset["train"].shuffle(seed=script_args.seed)
        # Accumulate valid examples until we reach top_k_train (or run out of data)
        valid_examples = accumulate_valid_examples(
            shuffled_train,
            lambda x: is_valid_language(x, key, expected_lang_code),
            train_max_size
        )
        if len(valid_examples) < train_max_size:
            logger.warning(f"Only {len(valid_examples)} valid examples found, fewer than the desired {train_max_size}.")
        # Create a dataset from the valid examples and format it for COMET
        valid_dataset = Dataset.from_list(valid_examples)
        valid_dataset = valid_dataset.map(lambda x: format_to_comet(x, language_pair), batched=True)
        
        logger.info(f"Selecting top sentences with {script_args.comet_model} model after LID filtering.")
        train = get_top_sentences(
            valid_dataset, model, script_args.batch_size, script_args.top_k_train, script_args.device, script_args.seed
        )
        comet_name = script_args.comet_model.split("/")[1]
        output_path = os.path.join(script_args.output_dir, dataset_name, comet_name, new_language_pair)
        os.makedirs(output_path, exist_ok=True)
        convert_json(train, script_args.dataset_name, os.path.join(output_path, f"train_{script_args.top_k_train}.json"), new_language_pair, language_pair, comet_name)
    
    # ---- Branch for random sampling when COMET-QE model is not provided ----
    elif script_args.dataset_name == "allenai/nllb" and script_args.comet_model == "None":
        logger.info("No COMET-QE model provided. Using random sampling with LID filtering.")
        comet_name = "random"
        output_path = os.path.join(script_args.output_dir, dataset_name, comet_name, new_language_pair)
        os.makedirs(output_path, exist_ok=True)

        shuffled_train = raw_dataset["train"].shuffle(seed=script_args.seed)
        valid_examples = accumulate_valid_examples(
            shuffled_train,
            lambda x: is_valid_language(x, key, expected_lang_code),
            script_args.top_k_train
        )
        if len(valid_examples) < script_args.top_k_train:
            logger.warning(f"Only {len(valid_examples)} valid examples found, fewer than the desired {script_args.top_k_train}.")
        valid_dataset = Dataset.from_list(valid_examples)
        valid_dataset = valid_dataset.map(lambda x: format_to_comet(x, language_pair), batched=True)
        convert_json(valid_dataset, script_args.dataset_name, os.path.join(output_path, f"train_{script_args.top_k_train}.json"), new_language_pair, language_pair, comet_name)
    
    # ---- Other datasets (e.g., flores, mafand) ----
    elif script_args.dataset_name in ["facebook/flores", "masakhane/mafand"]:
        comet_name = "random"
        output_path = os.path.join(script_args.output_dir, dataset_name, new_language_pair)
        os.makedirs(output_path, exist_ok=True)
        for split in raw_dataset.keys():
            convert_json(raw_dataset[split], script_args.dataset_name, os.path.join(output_path, f"{split}.json"), new_language_pair, language_pair, comet_name)
        if script_args.dataset_name == "masakhane/mafand":
            input_files = os.listdir(output_path)
            input_files = [os.path.join(output_path, file) for file in input_files]
            merge_jsonlines(input_files, os.path.join(output_path, "merged.json"))
    
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
