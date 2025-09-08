import torch
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from contextlib import nullcontext
import logging
from model import Nordic1BModel
from data_processor import NordicDataProcessor, get_batch
from config import NORDIC_1B_CONFIG, NORDIC_1B_LARGE_CONFIG

logger = logging.getLogger(__name__)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def evaluate_model(model, tokenizer, test_prompts, device, max_new_tokens=100):
    model.eval()
    results = {}
    with torch.no_grad():
        for lang, prompts in test_prompts.items():
            results[lang] = []
            selected_prompts = random.sample(prompts, min(3, len(prompts)))
            for prompt in selected_prompts:
                try:
                    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
                    generated = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50, top_p=0.9)
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    results[lang].append({
                        'prompt': prompt,
                        'generated': generated_text
                    })
                except Exception as e:
                    logger.error(f"Error generating for prompt '{prompt}': {e}")
                    continue
    model.train()
    return results

def train_model(data_dir, checkpoint_dir, resume, model_size='1b'):
    if model_size == '1b-large':
        config = NORDIC_1B_LARGE_CONFIG
        logger.info("Using 1B-Large configuration (~1.1B parameters)")
    else:
        config = NORDIC_1B_CONFIG
        logger.info("Using 1B configuration (~1.0B parameters)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Found {torch.cuda.device_count()} GPUs - will use DataParallel")
        use_multi_gpu = True
    else:
        use_multi_gpu = False
    
    processed_data_dir = Path("processed_data")
    if not (processed_data_dir / "train.bin").exists() or os.path.getsize(processed_data_dir / "train.bin") < 1024:
        logger.info("Preprocessed data missing or invalid. Processing Nordic FinePDFs...")
        processor = NordicDataProcessor(data_dir=data_dir, max_context_length=config["context_length"])
        tokens = processor.load_and_tokenize_files(max_docs_per_lang=50000)  # Further reduced for faster testing
        train_tokens, val_tokens = processor.create_train_val_split(tokens)
        processor.save_tokenized_data(train_tokens, val_tokens)
        logger.info("Data preprocessing complete!")
    else:
        logger.info("Using existing preprocessed data.")
    
    logger.info(f"Initializing {model_size.upper()} Nordic model...")
    model = Nordic1BModel(config).to(device)
    
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
        logger.info("Enabled DataParallel for multi-GPU training")
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters - Total: {total_params:,} ({total_params/1e9:.2f}B), Trainable: {trainable_params:,}")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    processor = NordicDataProcessor()
    
    learning_rate = 2e-4
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    
    warmup_steps = 1500
    max_steps = 10000  
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=max_steps, 
        pct_start=warmup_steps/max_steps, anneal_strategy='cos'
    )
    
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda") if device == "cuda" else nullcontext()
    
    batch_size = 6 if use_multi_gpu else 4
    block_size = config["context_length"]
    gradient_accumulation_steps = 8 if use_multi_gpu else 12
    eval_interval = 1500
    save_interval = 4000
    
    logger.info(f"Training configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Block size: {block_size}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps * (2 if use_multi_gpu else 1)}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Max steps: {max_steps}")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    
    start_step = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if resume and checkpoint_file.exists():
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if use_multi_gpu:
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        logger.info(f"Resumed from step {start_step}")
    
    logger.info(f"Starting training from step {start_step} to {max_steps}")
    model.train()
    
    for step in tqdm(range(start_step, max_steps), initial=start_step, total=max_steps):
        X, Y = get_batch("train", data_dir="processed_data", block_size=block_size, batch_size=batch_size, device=device)
        
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        if step % eval_interval == 0 and step > 0:
            model.eval()
            train_loss_accum = 0
            val_loss_accum = 0
            eval_steps = 50
            
            for _ in range(eval_steps):
                X_train, Y_train = get_batch("train", data_dir="processed_data", 
                                           block_size=block_size, batch_size=batch_size, device=device)
                with ctx:
                    _, loss_train = model(X_train, Y_train)
                train_loss_accum += loss_train.item()
                
                X_val, Y_val = get_batch("val", data_dir="processed_data", 
                                       block_size=block_size, batch_size=batch_size, device=device)
                with ctx:
                    _, loss_val = model(X_val, Y_val)
                val_loss_accum += loss_val.item()
            
            train_loss_avg = train_loss_accum / eval_steps
            val_loss_avg = val_loss_accum / eval_steps
            train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)
            
            logger.info(f"Step {step}: train_loss={train_loss_avg:.4f}, val_loss={val_loss_avg:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
            
            if step % (eval_interval * 2) == 0:
                logger.info("="*60)
                logger.info("GENERATING SAMPLE TEXT:")
                logger.info("="*60)
                eval_model = model.module if use_multi_gpu else model
                results = evaluate_model(eval_model, tokenizer, processor.test_prompts, device, max_new_tokens=100)
                for lang, lang_results in results.items():
                    logger.info(f"\n--- {lang.upper()} ---")
                    for i, result in enumerate(lang_results):
                        logger.info(f"Prompt {i+1}: {result['prompt']}")
                        logger.info(f"Generated: {result['generated'][len(result['prompt']):].strip()}")
                        logger.info("")
                logger.info("="*60)
            
            model.train()
            
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_path = checkpoint_dir / "best_model.pt"
                model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
                torch.save({
                    "model": model_state,
                    "config": config,
                    "step": step,
                    "val_loss": val_loss_avg
                }, best_model_path)
                logger.info(f"Saved new best model with val_loss={val_loss_avg:.4f}")
        
        if step % save_interval == 0 and step > 0:
            model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
            checkpoint_data = {
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "config": config
            }
            if scaler:
                checkpoint_data["scaler"] = scaler.state_dict()
            torch.save(checkpoint_data, checkpoint_file)
            logger.info(f"Saved checkpoint at step {step}")
    
    logger.info("Training completed! Performing final evaluation...")
    final_model_path = checkpoint_dir / "final_model.pt"
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
    torch.save({
        "model": model_state,
        "config": config,
        "step": max_steps,
        "train_losses": train_losses,
        "val_losses": val_losses
    }, final_model_path)
    
    logger.info("\nFINAL EVALUATION:")
    logger.info("="*80)
    eval_model = model.module if use_multi_gpu else model
    final_results = evaluate_model(eval_model, tokenizer, processor.test_prompts, device, max_new_tokens=150)
    for lang, lang_results in final_results.items():
        logger.info(f"\n{'='*20} {lang.upper()} FINAL RESULTS {'='*20}")
        for i, result in enumerate(lang_results):
            logger.info(f"\nPrompt {i+1}: {result['prompt']}")
            logger.info(f"Generated: {result['generated'][len(result['prompt']):].strip()}")
    
    if train_losses and val_losses:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        steps_plot = [i * eval_interval for i in range(len(train_losses))]
        plt.plot(steps_plot, train_losses, 'b-', label='Train Loss', alpha=0.8)
        plt.plot(steps_plot, val_losses, 'r-', label='Val Loss', alpha=0.8)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(steps_plot, val_losses, 'r-', label='Val Loss', alpha=0.8)
        plt.xlabel('Training Steps')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(checkpoint_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    logger.info(f"\nTraining complete! Models saved to {checkpoint_dir}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i} - Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")