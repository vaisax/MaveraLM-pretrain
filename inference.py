import torch
from pathlib import Path
import logging
from model import Nordic1BModel, Nordic4BModel
from config import NORDIC_1B_CONFIG, NORDIC_1B_LARGE_CONFIG
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class NordicModelInference:
    def __init__(self, model_path, config=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = config or checkpoint.get("config", NORDIC_1B_CONFIG)
        
        # Determine model type based on config
        if self.config.get("emb_dim", 0) > 2000:  # 4B model has emb_dim 2816
            self.model = Nordic4BModel(self.config).to(self.device)
            logger.info("Loaded Nordic 4B model")
        else:  # 1B models have smaller emb_dim
            self.model = Nordic1BModel(self.config).to(self.device)
            logger.info("Loaded Nordic 1B model")
        
        # Load model weights
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        logger.info(f"Context length: {self.config['context_length']:,}")
        logger.info(f"Embedding dim: {self.config['emb_dim']}")

    def generate_text(self, prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text continuation for a given prompt"""
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Ensure input doesn't exceed context length
            if input_ids.size(1) > self.config["context_length"]:
                input_ids = input_ids[:, -self.config["context_length"]:]
                logger.warning(f"Input truncated to {self.config['context_length']} tokens")
            
            generated_ids = self.model.generate(
                input_ids, 
                max_new_tokens=max_length, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text

def interactive_chat(checkpoint_dir):
    """Start an interactive chat session with the model"""
    # Try to find a model file
    model_path = None
    checkpoint_dir = Path(checkpoint_dir)
    
    for filename in ["best_model.pt", "final_model.pt", "checkpoint.pt"]:
        potential_path = checkpoint_dir / filename
        if potential_path.exists():
            model_path = potential_path
            break
    
    if not model_path:
        logger.error("No trained model found. Please train the model first.")
        logger.error(f"Looking for models in: {checkpoint_dir}")
        return
    
    try:
        inference = NordicModelInference(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    logger.info("Starting interactive Nordic model chat...")
    logger.info("Type 'quit' to exit, 'help' for commands")
    
    while True:
        try:
            user_input = input("\nPrompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  quit/exit/q - Exit the chat")
                print("  help - Show this help")
                print("  settings - Show current generation settings")
                print("  Just type any text to generate a continuation")
                continue
            elif user_input.lower() == 'settings':
                print(f"Current settings:")
                print(f"  Model: {inference.config.get('emb_dim', 'unknown')} embedding dim")
                print(f"  Context length: {inference.config['context_length']:,}")
                print(f"  Device: {inference.device}")
                continue
            elif not user_input:
                continue
            
            print("\nGenerating...")
            response = inference.generate_text(
                user_input, 
                max_length=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            continuation = response[len(user_input):].strip()
            print(f"\nContinuation: {continuation}")
            
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")

def test_model(checkpoint_dir):
    """Test the model with predefined prompts"""
    # Try to find a model file
    model_path = None
    checkpoint_dir = Path(checkpoint_dir)
    
    for filename in ["best_model.pt", "final_model.pt", "checkpoint.pt"]:
        potential_path = checkpoint_dir / filename
        if potential_path.exists():
            model_path = potential_path
            break
    
    if not model_path:
        logger.error("No trained model found. Please train the model first.")
        return
    
    try:
        inference = NordicModelInference(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    test_cases = {
        'Swedish': [
            "Artificiell intelligens kommer att förändra",
            "I framtiden kommer svenska företag att",
            "Klimatförändringarna påverkar Sverige genom att",
            "Stockholm är en vacker stad som",
            "Under den kalla vintern i Sverige"
        ],
        'Danish': [
            "Kunstig intelligens vil ændre",
            "I fremtiden vil danske virksomheder",
            "Klimaforandringerne påvirker Danmark ved at",
            "København er en smuk by som",
            "I den kolde vinter i Danmark"
        ],
        'Norwegian': [
            "Kunstig intelligens vil endre",
            "I fremtiden vil norske selskaper",
            "Klimaendringene påvirker Norge ved at",
            "Oslo er en vakker by som",
            "Under den kalde vinteren i Norge"
        ]
    }
    
    print("\n" + "="*80)
    print(f"NORDIC 1B MODEL - INFERENCE TESTING")
    print(f"Model: {model_path.name}")
    print(f"Parameters: {sum(p.numel() for p in inference.model.parameters()):,}")
    print(f"Context length: {inference.config['context_length']:,}")
    print("="*80)
    
    for language, prompts in test_cases.items():
        print(f"\n{'='*25} {language.upper()} {'='*25}")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nTest {i}:")
            print(f"Prompt: {prompt}")
            
            try:
                generated = inference.generate_text(
                    prompt, 
                    max_length=100, 
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9
                )
                continuation = generated[len(prompt):].strip()
                print(f"Generated: {continuation}")
            except Exception as e:
                print(f"Error: {e}")
            
            print("-" * 70)
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Quick test if run directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="nordic_1b_checkpoints")
    parser.add_argument("--mode", choices=["test", "chat"], default="test")
    args = parser.parse_args()
    
    if args.mode == "test":
        test_model(args.checkpoint_dir)
    else:
        interactive_chat(args.checkpoint_dir)