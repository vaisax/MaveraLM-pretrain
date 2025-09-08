import torch
import numpy as np
import json
import gzip
import os
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class NordicDataProcessor:
    def __init__(self, data_dir="nordic_finepdfs", max_context_length=12000):
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
    
    def load_and_tokenize_files(self):
        logger.info("Loading and tokenizing Nordic FinePDFs data...")
        all_tokens = []
        total_docs = 0
        
        for file_path in self.data_dir.glob("*.jsonl.gz"):
            logger.info(f"Processing {file_path.name}...")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"Tokenizing {file_path.name}")):
                    try:
                        doc = json.loads(line.strip())
                        text = doc.get('text', '').strip()
                        if len(text) < 100:
                            continue
                        tokens = self.tokenizer.encode(text, add_special_tokens=True)
                        if len(tokens) > self.max_context_length:
                            overlap = 256
                            for i in range(0, len(tokens), self.max_context_length - overlap):
                                chunk = tokens[i:i + self.max_context_length]
                                if len(chunk) > 512:
                                    all_tokens.extend(chunk)
                                    all_tokens.append(self.tokenizer.eos_token_id)
                        else:
                            all_tokens.extend(tokens)
                            all_tokens.append(self.tokenizer.eos_token_id)
                        total_docs += 1
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num} in {file_path.name}: {e}")
                        continue
        logger.info(f"Processed {total_docs:,} documents, created {len(all_tokens):,} tokens")
        return np.array(all_tokens, dtype=np.uint32)
    
    def create_train_val_split(self, tokens, val_ratio=0.05):
        split_idx = int(len(tokens) * (1 - val_ratio))
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        logger.info(f"Train tokens: {len(train_tokens):,}")
        logger.info(f"Val tokens: {len(val_tokens):,}")
        return train_tokens, val_tokens
    
    def save_tokenized_data(self, train_tokens, val_tokens, output_dir="processed_data"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        train_path = output_dir / "train.bin"
        train_mmap = np.memmap(train_path, dtype=np.uint32, mode='w+', shape=(len(train_tokens),))
        train_mmap[:] = train_tokens
        train_mmap.flush()
        val_path = output_dir / "val.bin"
        val_mmap = np.memmap(val_path, dtype=np.uint32, mode='w+', shape=(len(val_tokens),))
        val_mmap[:] = val_tokens
        val_mmap.flush()
        logger.info(f"Saved tokenized data to {output_dir}")
        return train_path, val_path

def get_batch(split, data_dir="processed_data", block_size=6144, batch_size=4, device='cuda'):
    data_path = Path(data_dir) / f"{split}.bin"
    data = np.memmap(data_path, dtype=np.uint32, mode='r')
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y