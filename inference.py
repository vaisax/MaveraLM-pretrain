import torch
from pathlib import Path
import logging
from model import Nordic4BModel
from config import NORDIC_4B_CONFIG
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class NordicModelInference:
    def __init__(self, model_path, config=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = config or checkpoint.get("config", NORDIC_4B_CONFIG)
        self.model = Nordic4BModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        logger.info(f"Loaded Nordic model from {model_path}")

    def generate_text(self, prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.9):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            generated_ids = self.model.generate(input_ids, max_new_tokens=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text

def interactive_chat(checkpoint_dir):
    model_path = Path(checkpoint_dir) / "best_model.pt"
    if not model_path.exists():
        model_path = Path(checkpoint_dir) / "final_model.pt"
    if not model_path.exists():
        logger.error("No trained model found. Please train the model first.")
        return
    inference = NordicModelInference(str(model_path))
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
                print("  Just type any text to generate a continuation")
                continue
            elif not user_input:
                continue
            print("\nGenerating...")
            response = inference.generate_text(user_input, max_length=150)
            continuation = response[len(user_input):].strip()
            print(f"\nContinuation: {continuation}")
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")

def test_model(checkpoint_dir):
    model_path = Path(checkpoint_dir) / "best_model.pt"
    if not model_path.exists():
        model_path = Path(checkpoint_dir) / "final_model.pt"
    if not model_path.exists():
        logger.error("No trained model found. Please train the model first.")
        return
    inference = NordicModelInference(str(model_path))
    test_cases = {
        'Swedish': [
            "Artificiell intelligens kommer att förändra",
            "I framtiden kommer svenska företag att",
            "Klimatförändringarna påverkar Sverige genom att"
        ],
        'Danish': [
            "Kunstig intelligens vil ændre",
            "I fremtiden vil danske virksomheder",
            "Klimaforandringerne påvirker Danmark ved at"
        ],
        'Norwegian': [
            "Kunstig intelligens vil endre",
            "I fremtiden vil norske selskaper",
            "Klimaendringene påvirker Norge ved at"
        ]
    }
    print("\n" + "="*80)
    print("NORDIC 4B MODEL - INFERENCE TESTING")
    print("="*80)
    for language, prompts in test_cases.items():
        print(f"\n{'='*20} {language.upper()} {'='*20}")
        for i, prompt in enumerate(prompts, 1):
            print(f"\nTest {i}:")
            print(f"Prompt: {prompt}")
            try:
                generated = inference.generate_text(prompt, max_length=100, temperature=0.7)
                continuation = generated[len(prompt):].strip()
                print(f"Generated: {continuation}")
            except Exception as e:
                print(f"Error: {e}")
            print("-" * 60)