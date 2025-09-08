import torch
import numpy as np
import json
import gzip
import os
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

class NordicDataProcessor:
    def __init__(self, data_dir="nordic_finepdfs", max_context_length=8192):
        self.data_dir = Path(data_dir)
        self.max_context_length = max_context_length
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.test_prompts = {
            'swedish': [
                "Sverige är ett land i Norden som",
                "Stockholms universitet grundades",
                "I den svenska skogen finns det",
                "Den svenska regeringen beslutade att",
                "Nobelpriset delas ut årligen i",
                "Kungliga biblioteket i Stockholm",
                "Svenska företag som IKEA och Volvo",
                "Under vintern i Lappland",
                "Svenska språket tillhör",
                "Midsommar firas traditionellt"
            ],
            'danish': [
                "Danmark er et land i Skandinavien som",
                "Københavns universitet blev grundlagt",
                "I den danske hovedstad findes",
                "Den danske regering besluttede at",
                "H.C. Andersen skrev mange",
                "Det danske kongehus",
                "Danske virksomheder som Maersk",
                "Om vinteren i Danmark",
                "Det danske sprog",
                "Traditionen med hygge"
            ],
            'norwegian': [
                "Norge er et land i Skandinavia som",
                "Universitetet i Oslo ble grunnlagt",
                "I de norske fjordene",
                "Den norske regjeringen bestemte at",
                "Edvard Grieg komponerte",
                "Det norske kongehus",
                "Norske selskaper som Statoil",
                "Om vinteren i Norge",
                "Det norske språk",
                "Oljefondet i Norge"
            ]
        }
    
    def load_and_tokenize_files(self, max_docs_per_lang=10000):
        logger.info("Loading and tokenizing Nordic FinePDFs data...")
        all_tokens = []
        total_docs = 0
        
        # Try Parquet first
        parquet_dir = self.data_dir / "parquet"
        parquet_files = list(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else []
        if parquet_files:
            logger.info(f"Found {len(parquet_files)} parquet files in {parquet_dir}, processing...")
            all_tokens, total_docs = self._load_from_parquet(parquet_files, max_docs_per_lang)
        
        # Fallback to JSONL
        if not all_tokens:
            logger.info("No valid parquet files processed, trying JSONL...")
            jsonl_files = list(self.data_dir.glob("*.jsonl.gz"))
            if not jsonl_files:
                logger.warning(f"No JSONL files found in {self.data_dir}/")
            else:
                for file_path in jsonl_files:
                    logger.info(f"Processing {file_path.name}...")
                    try:
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            doc_count = 0
                            for line_num, line in enumerate(tqdm(f, desc=f"Tokenizing {file_path.name}")):
                                if doc_count >= max_docs_per_lang:
                                    break
                                try:
                                    doc = json.loads(line.strip())
                                    text = doc.get('text', '').strip()
                                    
                                    if len(text) < 50:  # Relaxed from 100
                                        logger.debug(f"Skipping short text (len={len(text)}) in {file_path.name}")
                                        continue
                                    
                                    tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_context_length, truncation=True)
                                    
                                    if len(tokens) > self.max_context_length:
                                        overlap = 256
                                        for i in range(0, len(tokens), self.max_context_length - overlap):
                                            chunk = tokens[i:i + self.max_context_length]
                                            if len(chunk) > 256:  # Relaxed from 512
                                                all_tokens.extend(chunk)
                                                all_tokens.append(self.tokenizer.eos_token_id)
                                    else:
                                        all_tokens.extend(tokens)
                                        all_tokens.append(self.tokenizer.eos_token_id)
                                    
                                    total_docs += 1
                                    doc_count += 1
                                    
                                    if total_docs % 1000 == 0:
                                        logger.info(f"Processed {total_docs:,} documents, {len(all_tokens):,} tokens")
                                    
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid JSON in {file_path.name}, line {line_num}")
                                    continue
                                except Exception as e:
                                    logger.warning(f"Error processing line {line_num} in {file_path.name}: {e}")
                                    continue
                    except Exception as e:
                        logger.error(f"Failed to read {file_path}: {e}")
                        continue
        
        # Fallback to streaming if still empty
        if not all_tokens:
            logger.info("No local data processed, falling back to streaming from Hugging Face...")
            all_tokens, total_docs = self._load_from_hf_streaming(max_docs_per_lang=max_docs_per_lang)
        
        if not all_tokens:
            raise ValueError(f"No tokens generated - check input data in {self.data_dir}/ or streaming connection")
        
        logger.info(f"Processed {total_docs:,} documents, created {len(all_tokens):,} tokens")
        return np.array(all_tokens, dtype=np.uint32)
    
    def _load_from_parquet(self, parquet_files, max_docs_per_lang):
        import pandas as pd
        all_tokens = []
        total_docs = 0
        
        for file_path in parquet_files:
            logger.info(f"Processing {file_path.name}...")
            try:
                df = pd.read_parquet(file_path)
                doc_count = 0
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {file_path.name}"):
                    if doc_count >= max_docs_per_lang:
                        break
                    try:
                        text = row.get('text', '').strip()
                        if len(text) < 50:
                            logger.debug(f"Skipping short text (len={len(text)}) in {file_path.name}")
                            continue
                        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_context_length, truncation=True)
                        if len(tokens) > self.max_context_length:
                            overlap = 256
                            for i in range(0, len(tokens), self.max_context_length - overlap):
                                chunk = tokens[i:i + self.max_context_length]
                                if len(chunk) > 256:
                                    all_tokens.extend(chunk)
                                    all_tokens.append(self.tokenizer.eos_token_id)
                        else:
                            all_tokens.extend(tokens)
                            all_tokens.append(self.tokenizer.eos_token_id)
                        total_docs += 1
                        doc_count += 1
                    except Exception as e:
                        logger.warning(f"Error processing row {idx} in {file_path.name}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Failed to read parquet file {file_path}: {e}")
                continue
        
        logger.info(f"Processed {total_docs:,} documents from parquet, created {len(all_tokens):,} tokens")
        return all_tokens, total_docs
    
    def _load_from_hf_streaming(self, languages=['dan_Latn', 'swe_Latn', 'nob_Latn', 'isl_Latn'], max_docs_per_lang=10000):
        all_tokens = []
        total_docs = 0
        for lang in languages:
            logger.info(f"Streaming {lang} from Hugging Face (max {max_docs_per_lang} docs)...")
            try:
                ds = load_dataset("HuggingFaceFW/finepdfs", name=lang, split="train", streaming=True)
                doc_count = 0
                for row in ds:
                    if doc_count >= max_docs_per_lang:
                        break
                    try:
                        text = row.get('text', '').strip()
                        if len(text) < 50:
                            continue
                        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_context_length, truncation=True)
                        if len(tokens) > self.max_context_length:
                            overlap = 256
                            for i in range(0, len(tokens), self.max_context_length - overlap):
                                chunk = tokens[i:i + self.max_context_length]
                                if len(chunk) > 256:
                                    all_tokens.extend(chunk)
                                    all_tokens.append(self.tokenizer.eos_token_id)
                        else:
                            all_tokens.extend(tokens)
                            all_tokens.append(self.tokenizer.eos_token_id)
                        total_docs += 1
                        doc_count += 1
                    except Exception as e:
                        logger.warning(f"Error in {lang} row: {e}")
                        continue
            except Exception as e:
                logger.error(f"Failed to stream {lang}: {e}")
                continue
        logger.info(f"Streamed {total_docs:,} documents, created {len(all_tokens):,} tokens")
        return all_tokens, total_docs
    
    def create_train_val_split(self, tokens, val_ratio=0.05):
        if len(tokens) == 0:
            raise ValueError("No tokens to split - check input data!")
        split_idx = int(len(tokens) * (1 - val_ratio))
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        logger.info(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")
        return train_tokens, val_tokens
    
    def save_tokenized_data(self, train_tokens, val_tokens, output_dir="processed_data"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for tokens, name in [(train_tokens, "train"), (val_tokens, "val")]:
            if len(tokens) == 0:
                raise ValueError(f"{name} tokens array is empty - check preprocessing!")
            if not isinstance(tokens, np.ndarray) or tokens.dtype != np.uint32:
                tokens = np.array(tokens, dtype=np.uint32)
        
        train_path = output_dir / "train.bin"
        train_mmap = np.memmap(train_path, dtype=np.uint32, mode='w+', shape=(len(train_tokens),))
        train_mmap[:] = train_tokens
        train_mmap.flush()
        
        val_path = output_dir / "val.bin"
        val_mmap = np.memmap(val_path, dtype=np.uint32, mode='w+', shape=(len(val_tokens),))
        val_mmap[:] = val_tokens
        val_mmap.flush()
        
        for path, length, name in [(train_path, len(train_tokens), "train"), (val_path, len(val_tokens), "val")]:
            file_size = os.path.getsize(path)
            expected_size = length * 4
            if file_size != expected_size:
                raise ValueError(f"{name} file size ({file_size}) != expected ({expected_size})")
            logger.info(f"Saved {length:,} {name} tokens ({file_size} bytes)")
        
        metadata = {
            "vocab_size": len(self.tokenizer),
            "max_context_length": self.max_context_length,
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
            "tokenizer": "microsoft/DialoGPT-medium"
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return train_path, val_path

def get_batch(split, data_dir="processed_data", block_size=8192, batch_size=4, device='cuda'):
    data_path = Path(data_dir) / f"{split}.bin"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_path} does not exist - run preprocessing!")
    file_size = os.path.getsize(data_path)
    if file_size == 0 or file_size % 4 != 0:
        raise ValueError(f"Invalid {data_path} size ({file_size} bytes) - delete and reprocess!")
    data = np.memmap(data_path, dtype=np.uint32, mode='r')
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"Dataset too small for block_size {block_size}")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y