import argparse
import logging
from train import train_model
from inference import test_model, interactive_chat

# Setup logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Nordic FinePDFs 1B Model')
    parser.add_argument('--mode', choices=['train', 'test', 'chat'], default='train',
                       help='Mode: train the model, test inference, or interactive chat')
    parser.add_argument('--data-dir', default='../nordic_finepdfs',
                       help='Directory containing Nordic FinePDFs data')
    parser.add_argument('--checkpoint-dir', default='nordic_1b_checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--model-size', choices=['1b', '1b-large'], default='1b',
                       help='Model size variant: 1b (~1.0B params) or 1b-large (~1.1B params)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        logger.info(f"Starting Nordic {args.model_size.upper()} model training...")
        train_model(args.data_dir, args.checkpoint_dir, args.resume, args.model_size)
    elif args.mode == 'test':
        logger.info(f"Testing Nordic {args.model_size.upper()} model inference...")
        test_model(args.checkpoint_dir)
    elif args.mode == 'chat':
        logger.info("Starting interactive chat...")
        interactive_chat(args.checkpoint_dir)

if __name__ == "__main__":
    main()