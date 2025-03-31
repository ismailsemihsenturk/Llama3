from model_iss_train import TinyShakespeareDataset
import os
import torch
from model_iss import Transformer, ModelArgs  # Adjust the import as needed
from bpe_tokenizer import BPETokenizerSimple  # Import BPE Tokenizer

TOKENIZER_VOCAB_PATH = "bpe_vocab.json"  # Define paths for vocab and merges
TOKENIZER_MERGES_PATH = "bpe_merges.txt"


def sample_from_model(model, device, max_length=100, prompt="The"):
    """
    Generates text from the trained model.

    Args:
        model: The trained Transformer model.
        stoi: Dictionary mapping characters to integers.
        itos: Dictionary mapping integers to characters.
        device: The device to run inference on.
        max_length: Maximum length of generated text.
        prompt: Seed text to start the generation.

    Returns:
        Generated text.
    """

    tokenizer = BPETokenizerSimple()  # Load tokenizer for inference
    tokenizer.load_vocab_and_merges(
        TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)
    # Convert prompt to tensor
    input_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(
        0).to(device)  # Encode prompt using BPE # Shape: [1, seq_len]

    # Set model to evaluation mode
    model.eval()

    # Generate characters one by one
    start_position = 0
    with torch.no_grad():
        for _ in range(max_length - start_position):
            # Forward pass
            # Shape: [1, seq_len, vocab_size]
            # Forward pass with only the most recent token
            # This assumes the model is designed to use the KV cache across calls.
            output = model(input_tokens[:, -1:], start_position)

            # Get the last token's predictions
            next_token_logits = output[:, -1, :]  # Shape: [1, vocab_size]

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            # Top-k ensures we avoid crazy, unlikely words.
            # Temperature adds controlled randomness.
            top_k = 10
            temperature = 0.7
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            top_probs = torch.nn.functional.softmax(
                top_probs / temperature, dim=-1)
            # Sample from the probability distribution
            next_token = top_indices[0, torch.multinomial(
                top_probs, num_samples=1)]
            next_token = next_token.squeeze(0).squeeze(
                0)  # Shape: [1], convert to 1D tensor

            # Append the predicted token to input, checking the maximum length
            if input_tokens.size(1) < args.max_seq_len:
                input_tokens = torch.cat(
                    [input_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            else:
                break
            start_position += 1  # Increment start_position

            # Stop if the model generates an EOS (End of Sentence) token (if applicable)
            decoded_token = tokenizer.decode([next_token.item()]) # Decode to string to check
            if decoded_token.startswith("."): # Check if decoded token starts with "." (might need refinement)
                print("EOS token ('.') generated. Stopping.")
                break

    # Convert tokens back to characters
    generated_text = tokenizer.decode(input_tokens.squeeze(
        0).tolist())  # Decode using BPE tokenizer

    return generated_text


if __name__ == "__main__":
    # Load dataset for vocab
    dataset = TinyShakespeareDataset('tiny_shakespeare.txt', seq_length=100)

    # GENERATE TEXT:
    # ----------------------------------------------------------------------------------------------------
    # Define ModelArgs *once*
    args = ModelArgs(vocab_size=dataset.vocab_size, max_seq_len=100,n_kv_heads=4,
                     device="cuda" if torch.cuda.is_available() else "cpu")  # vocab_size obtained from dataset

    transformer = Transformer(args).to(args.device)
    checkpoint_dir = 'checkpoints'
    # Load global step from file if it exists
    global_step_file = os.path.join(checkpoint_dir, 'global_step.txt')
    if os.path.exists(global_step_file):
        with open(global_step_file, 'r') as f:
            global_step = int(f.read())
        print(f"Resuming sampling from global step: {global_step}")
        # Load checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f'transformer_epoch_{global_step}.pth')
        transformer.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Starting sampling from scratch with random weights.")

    transformer.eval()  # Set the model to evaluation mode

    # Sample
    generated_text = sample_from_model(
        transformer, args.device, max_length=args.max_seq_len, prompt="First Citizen:")
    print("Generated Text:\n", generated_text)
# --------------------------------------------------------------------------------------------------------
