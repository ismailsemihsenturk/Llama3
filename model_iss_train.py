import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model_iss import Transformer, ModelArgs  # Adjust the import as needed
from bpe_tokenizer import BPETokenizerSimple  # Import BPE Tokenizer

TOKENIZER_VOCAB_PATH = "bpe_vocab.json"  # Define paths for vocab and merges
TOKENIZER_MERGES_PATH = "bpe_merges.txt"


class TinyShakespeareDataset(Dataset):
    def __init__(self, filepath, seq_length=20):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # Initialize and load BPE tokenizer
        self.tokenizer = BPETokenizerSimple()
        self.tokenizer.load_vocab_and_merges(
            TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)  # Load trained tokenizer
        self.vocab_size = len(self.tokenizer.vocab)  # BPE vocab size
        # self.tokenizer.inverse_vocab = BPE token to index
        # self.tokenizer.vocab  = Index to BPE token

        # Encode entire text to integer sequence
        self.data = torch.tensor(self.tokenizer.encode(
            self.text), dtype=torch.long)  # Encode text using BPE tokenizer

        self.seq_length = seq_length

    def __len__(self):
        # Total number of sequences that can be drawn from text
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # For input, take seq_length tokens, and for target, the next token.
        x = self.data[idx: idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]
        return x, y


def train_transformer(model, train_loader, val_loader, optimizer, criterion, args, num_epochs, checkpoint_dir='checkpoints'):
    """
    Trains the Transformer model and saves checkpoints based on global step.
    """
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.train()
    global_step = 0  # Initialize global step

    # Load global step from file if it exists
    global_step_file = os.path.join(checkpoint_dir, 'global_step.txt')
    if os.path.exists(global_step_file):
        with open(global_step_file, 'r') as f:
            global_step = int(f.read())
        print(f"Resuming training from global step: {global_step}")
        # Load checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f'transformer_epoch_{global_step}.pth')
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Starting training from scratch with random weights.")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs, start_position=0)
            loss = criterion(
                outputs.view(-1, args.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1  # Increment global step

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Train Loss: {loss.item():.4f}, Global Step: {global_step}")
                if loss.item() < 0.0001:
                    print("Early stopping! Loss is below 0.0001")
                    # Save checkpoint before early stopping
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f'transformer_epoch_{global_step}.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'Trained model saved to {checkpoint_path}')
                    # Save global step to file
                    with open(global_step_file, 'w') as f:
                        f.write(str(global_step))
                    return  # Exit the training loop

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Average Train Epoch Loss: {avg_epoch_loss:.4f}")

        # Validation Loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for validation
            for val_batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                outputs = model(inputs, start_position=0)
                loss = criterion(
                    outputs.view(-1, args.vocab_size), targets.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")
        model.train()  # Set model back to training mode

        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(
            checkpoint_dir, f'transformer_epoch_{global_step}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')
        # Save global step to file
        with open(global_step_file, 'w') as f:
            f.write(str(global_step))


def evaluate_transformer(model, test_loader, criterion, args):
    """
    Evaluates the Transformer model on the test set.
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        # Correctly unpack index and batch data
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(args.device)  # Now inputs is the input tensor
            targets = targets.to(args.device)  # targets is the target tensor
            outputs = model(inputs, start_position=0)
            loss = criterion(
                outputs.view(-1, args.vocab_size), targets.view(-1))
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    # TOKENIZATION:
    # ----------------------------------------------------------------------------------------------------
    # Before dataset creation, train the BPE tokenizer and save vocab and merges:
    # Check if tokenizer is already trained
    if not os.path.exists(TOKENIZER_VOCAB_PATH) or not os.path.exists(TOKENIZER_MERGES_PATH):
        print("Training BPE Tokenizer...")
        tokenizer = BPETokenizerSimple()
        # Load corpus for training tokenizer
        with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
            corpus = f.read()
        # Train with vocab_size=1000 (adjust as needed)
        tokenizer.train(corpus, vocab_size=1000)
        tokenizer.save_vocab_and_merges(
            TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)  # Save trained tokenizer
        print(
            f"BPE Tokenizer trained and saved to {TOKENIZER_VOCAB_PATH} and {TOKENIZER_MERGES_PATH}")
    else:
        print(
            f"Loading pre-trained BPE Tokenizer from {TOKENIZER_VOCAB_PATH} and {TOKENIZER_MERGES_PATH}")
    # ----------------------------------------------------------------------------------------------------

    # DATA:
    # ----------------------------------------------------------------------------------------------------
    dataset = TinyShakespeareDataset('tiny_shakespeare.txt', seq_length=100)

    # Limit the dataset size
    dataset.data = dataset.data[:50000]

    # Split the dataset into training, validation, and test sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))  # 10% for validation
    test_size = len(dataset) - train_size - \
        val_size  # Remaining 10% for testing

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each set
    batch_size = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    # No need to shuffle validation data
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # No need to shuffle test data
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    # -----------------------------------------------------------------------------------------------------

    # TRAIN AND EVAL:
    # -----------------------------------------------------------------------------------------------------
    args = ModelArgs(vocab_size=dataset.vocab_size, max_seq_len=100,n_kv_heads=4,
                     device="cuda" if torch.cuda.is_available() else "cpu")  # vocab_size obtained from dataset
    # max_seq_len is the sequence length of the dataset
    transformer = Transformer(args).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer.parameters(), lr=1e-3)
    num_epochs = 1
    checkpoint_dir = 'checkpoints'
    # Load global step from file if it exists
    global_step_file = os.path.join(checkpoint_dir, 'global_step.txt')
    if os.path.exists(global_step_file):
        with open(global_step_file, 'r') as f:
            global_step = int(f.read())
        print(f"Resuming training from global step: {global_step}")
        # Load checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f'transformer_epoch_{global_step}.pth')
        transformer.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Starting training from scratch with random weights.")
    train_transformer(transformer, train_loader, val_loader, optimizer,
                      criterion, args, num_epochs, checkpoint_dir)

    # Evaluate the model on the test set:
    evaluate_transformer(transformer, test_loader, criterion, args)
# --------------------------------------------------------------------------------------------------------
