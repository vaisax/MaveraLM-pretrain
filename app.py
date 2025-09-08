import argparse
import logging
from train import train_model
from inference import test_model, interactive_chat

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Nordic FinePDFs 4B Model')
    parser.add_argument('--mode', choices=['train', 'test', 'chat'], default='train',
                       help='Mode: train the model, test inference, or interactive chat')
    parser.add_argument('--data-dir', default='nordic_finepdfs',
                       help='Directory containing Nordic FinePDFs data')
    parser.add_argument('--checkpoint-dir', default='nordic_4b_checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        logger.info("Starting Nordic 4B model training...")
        train_model(args.data_dir, args.checkpoint_dir, args.resume)
    elif args.mode == 'test':
        logger.info("Testing Nordic 4B model inference...")
        test_model(args.checkpoint_dir)
    elif args.mode == 'chat':
        logger.info("Starting interactive chat...")
        interactive_chat(args.checkpoint_dir)

if __name__ == "__main__":
    main()