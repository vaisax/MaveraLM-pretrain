# MaveraLM-4B

MaveraLM-4B is a 4-billion parameter transformer-based language model designed for processing and generating text in Nordic languages (Swedish, Danish, Norwegian). It is trained on a dataset of Nordic FinePDFs, supporting a context length of 12,000 tokens, 50B tokens. The model uses grouped query attention (GQA), SwiGLU feed-forward layers, RMSNorm, and rotary position embeddings (RoPE) for efficient and high-performance language modeling.

## Project Structure

- `app.py`: Main entry point for running the model in different modes (train, test, chat).
- `config.py`: Model configuration parameters for MaveraLM-4B.
- `data_processor.py`: Handles data preprocessing and tokenization of Nordic FinePDFs.
- `model.py`: Defines the transformer model architecture.
- `train.py`: Implements the training loop and evaluation logic.
- `inference.py`: Provides utilities for model inference and interactive chat.
- `utils.py`: Contains helper functions for performance analysis and visualization.
- `nordic_finepdfs/`: Directory for input data (not included in repository).
- `processed_data/`: Directory for preprocessed tokenized data (not included).
- `nordic_4b_checkpoints/`: Directory for model checkpoints (not included).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vaisax/MaveraLM-pretrain
   cd MaveraLM-pretrain
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy transformers tqdm matplotlib
   ```

3. Prepare the data:
   - Place your Nordic FinePDFs dataset (`.jsonl.gz` files) in the `nordic_finepdfs/` directory.

## Usage

Run the model in one of three modes:

- **Training**:
  ```bash
  python app.py --mode train --data-dir nordic_finepdfs
  ```
  - Use `--resume` to continue training from the latest checkpoint:
    ```bash
    python app.py --mode train --resume
    ```

- **Testing**:
  ```bash
  python app.py --mode test
  ```

- **Interactive Chat**:
  ```bash
  python app.py --mode chat
  ```

Checkpoints are saved in `nordic_4b_checkpoints/`, and preprocessed data is stored in `processed_data/`.

## Model Details

- **Parameters**: ~4 billion (exact count: ~4.01B)
- **Context Length**: 12,000 tokens
- **Architecture**:
  - Embedding dimension: 2,816
  - Layers: 32
  - Attention heads: 22
  - KV groups: 11 (for grouped query attention)
  - Feed-forward hidden dimension: 11,264
  - Normalization: RMSNorm
  - Activation: SwiGLU
  - Positional encoding: RoPE
  - Precision: bfloat16

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Transformers (Hugging Face)
- TQDM
- Matplotlib

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.

## Contact

For questions, please open an issue on GitHub.