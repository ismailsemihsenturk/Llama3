from collections import Counter, deque
from functools import lru_cache
import json
import os
import urllib.request

#region BPE Explanation
# Corpus: corpus = "ababcc"
# Desired Vocabulary Size: vocab_size = 5

# We will initialize a BPETokenizerSimple instance and call the train method:

# tokenizer = BPETokenizerSimple()
# tokenizer.train(corpus, vocab_size=5)
# content_copy
# download
# Use code with caution.
# Python

# Let's go inside the train method step-by-step:

# 1. Initialization inside train(self, text, vocab_size, allowed_special):

# text becomes "ababcc"

# vocab_size is 5

# allowed_special is {"<|endoftext|>"} (default)

# processed_text = [] (empty list)

# 2. Preprocessing (Space Handling):

# processed_text = []
# for i, char in enumerate(text):
#     if char == " " and i != 0:
#         processed_text.append("Ġ")
#     if char != " ":
#         processed_text.append(char)
# processed_text = "".join(processed_text)
# content_copy
# download
# Use code with caution.
# Python

# For text = "ababcc":

# The loop iterates through "ababcc".

# There are no spaces (except maybe at index 0, which is not handled here as the condition i != 0 will be false for the first character).

# So, processed_text becomes ['a', 'b', 'a', 'b', 'c', 'c'] and then joined into string "ababcc".

# processed_text is now "ababcc".

# 3. Initialize Vocabulary with Base Characters:

# unique_chars = [chr(i) for i in range(256)]
# unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)
# if "Ġ" not in unique_chars:
#     unique_chars.append("Ġ")

# self.vocab = {i: char for i, char in enumerate(unique_chars)}
# self.inverse_vocab = {char: i for i, char in self.vocab.items()}
# content_copy
# download
# Use code with caution.
# Python

# unique_chars starts as a list of the first 256 ASCII characters.

# set(processed_text) is {'a', 'b', 'c'}.

# It extends unique_chars with 'a', 'b', 'c' if they are not already in unique_chars (they won't be in the first 256 ASCII, except maybe 'a', 'b', 'c' themselves if they are within ASCII range, but this part ensures all unique chars from processed_text are included).

# "Ġ" is checked, and if not present (it's unlikely to be in "ababcc" itself), it's appended to unique_chars. Let's assume "Ġ" is added.

# self.vocab is created mapping indices to unique_chars. The first entries will be ASCII characters, followed by 'a', 'b', 'c', 'Ġ' (if added). Let's say 'a', 'b', 'c' are within the first 256 ASCII and 'Ġ' gets a later index. For simplicity, let's assume initial relevant part of self.vocab and self.inverse_vocab looks like (indices are just examples):

# self.vocab = {
#     ... , 97: 'a', 98: 'b', 99: 'c', ..., 256: 'Ġ'
# }
# self.inverse_vocab = {
#     ... , 'a': 97, 'b': 98, 'c': 99, ..., 'Ġ': 256
# }
# content_copy
# download
# Use code with caution.

# (In reality, the initial 256 chars would be ASCII control chars, punctuation, digits, letters, etc. 'a', 'b', 'c' are likely within the first 256, and 'Ġ' would be added at the end if not already present in the first 256). For simplicity let's assume 'a' gets ID 0, 'b' gets ID 1, 'c' gets ID 2, and 'Ġ' gets ID 3 initially for this example trace, ignoring actual ASCII values for now.

# self.vocab = {0: 'a', 1: 'b', 2: 'c', 3: 'Ġ'}
# self.inverse_vocab = {'a': 0, 'b': 1, 'c': 2, 'Ġ': 3}
# content_copy
# download
# Use code with caution.

# 4. Add Allowed Special Tokens:

# if allowed_special:
#     for token in allowed_special:
#         if token not in self.inverse_vocab:
#             new_id = len(self.vocab)
#             self.vocab[new_id] = token
#             self.inverse_vocab[token] = new_id
# content_copy
# download
# Use code with caution.
# Python

# allowed_special is {"<|endoftext|>"}.

# "<|endoftext|>" is checked if it's in self.inverse_vocab. It's not (initially only 'a', 'b', 'c', 'Ġ' are).

# new_id = len(self.vocab) is 4 (current vocab size).

# self.vocab[4] = "<|endoftext|>"

# self.inverse_vocab["<|endoftext|>"] = 4

# Updated self.vocab and self.inverse_vocab (example):

# self.vocab = {0: 'a', 1: 'b', 2: 'c', 3: 'Ġ', 4: '<|endoftext|>'}
# self.inverse_vocab = {'a': 0, 'b': 1, 'c': 2, 'Ġ': 3, '<|endoftext|>': 4}
# content_copy
# download
# Use code with caution.

# 5. Initial Tokenization:

# token_ids = [self.inverse_vocab[char] for char in processed_text]
# content_copy
# download
# Use code with caution.
# Python

# processed_text is "ababcc".

# It converts each character to its token ID using self.inverse_vocab.

# token_ids becomes [0, 1, 0, 1, 2, 2] (because 'a': 0, 'b': 1, 'c': 2).

# 6. BPE Merging Iterations (Loop starts, vocab_size = 5, current vocab size is 5):

# for new_id in range(len(self.vocab), vocab_size): # range(5, 5) - loop will not execute as range is empty
#     ...
# content_copy
# download
# Use code with caution.
# Python

# range(len(self.vocab), vocab_size) is range(5, 5), which is an empty range.

# The for loop will not execute even once because the initial vocabulary size is already 5, which is equal to the desired vocab_size.

# In this specific example with corpus = "ababcc" and vocab_size = 5, no BPE merges will happen because the initial vocabulary (characters + special tokens) already reached the target vocabulary size before the merging loop could even start.

# Let's change vocab_size to something larger, like vocab_size = 7, to see merges in action.

# Retrying with corpus = "ababcc" and vocab_size = 7

# Let's restart from step 6 with vocab_size = 7 and the same initial state up to step 5:

# self.vocab = {0: 'a', 1: 'b', 2: 'c', 3: 'Ġ', 4: '<|endoftext|>'}, self.inverse_vocab = {'a': 0, 'b': 1, 'c': 2, 'Ġ': 3, '<|endoftext|>': 4}

# token_ids = [0, 1, 0, 1, 2, 2]

# 6. BPE Merging Iterations (Loop starts, vocab_size = 7, current vocab size is 5):

# for new_id in range(len(self.vocab), vocab_size): # range(5, 7) - loop will execute for new_id = 5, 6
#     pair_id = self.find_freq_pair(token_ids, mode="most")
#     if pair_id is None:
#         break
#     token_ids = self.replace_pair(token_ids, pair_id, new_id)
#     self.bpe_merges[pair_id] = new_id
# content_copy
# download
# Use code with caution.
# Python

# Iteration 1: new_id = 5

# pair_id = self.find_freq_pair(token_ids, mode="most"):

# token_ids is [0, 1, 0, 1, 2, 2]. Pairs are [(0, 1), (1, 0), (0, 1), (1, 2), (2, 2)].

# Counter(pairs) is Counter({(0, 1): 2, (1, 0): 1, (1, 2): 1, (2, 2): 1}).

# Most frequent pair is (0, 1) (frequency 2).

# pair_id becomes (0, 1).

# if pair_id is None: break: pair_id is not None.

# token_ids = self.replace_pair(token_ids, pair_id, new_id):

# token_ids (input) is [0, 1, 0, 1, 2, 2]. pair_id is (0, 1), new_id is 5.

# replace_pair replaces all (0, 1) pairs with 5.

# token_ids becomes [5, 5, 2, 2] (pairs at index 0-1 and 2-3 are replaced).

# self.bpe_merges[pair_id] = new_id:

# self.bpe_merges[(0, 1)] = 5. We record that pair (0, 1) is merged to token ID 5.

# Loop continues to next iteration as vocabulary size (now 6 after adding merge) is still less than 7.

# Iteration 2: new_id = 6

# pair_id = self.find_freq_pair(token_ids, mode="most"):

# token_ids is now [5, 5, 2, 2]. Pairs are [(5, 5), (5, 2), (2, 2)].

# Counter(pairs) is Counter({(5, 5): 1, (5, 2): 1, (2, 2): 1}).

# All pairs have frequency 1. Let's assume max() picks the first one it encounters in the Counter's items, which might be (5, 5). (Tie-breaking can be arbitrary in this implementation).

# pair_id becomes (5, 5).

# if pair_id is None: break: pair_id is not None.

# token_ids = self.replace_pair(token_ids, pair_id, new_id):

# token_ids (input) is [5, 5, 2, 2]. pair_id is (5, 5), new_id is 6.

# replace_pair replaces (5, 5) with 6.

# token_ids becomes [6, 2, 2] (pair at index 0-1 is replaced).

# self.bpe_merges[pair_id] = new_id:

# self.bpe_merges[(5, 5)] = 6. We record that pair (5, 5) is merged to token ID 6.

# Loop continues to next iteration as vocabulary size (now 7) is not less than 7 anymore.

# Iteration 3: new_id = 7

# range(len(self.vocab), vocab_size) is now range(7, 7), which is empty. The loop terminates.

# 7. Build Vocabulary with Merged Tokens:

# for (p0, p1), new_id in self.bpe_merges.items():
#     merged_token = self.vocab[p0] + self.vocab[p1]
#     self.vocab[new_id] = merged_token
#     self.inverse_vocab[merged_token] = new_id
# content_copy
# download
# Use code with caution.
# Python

# self.bpe_merges is {(0, 1): 5, (5, 5): 6}.

# For (0, 1): 5:

# p0 = 0, p1 = 1, new_id = 5.

# merged_token = self.vocab[0] + self.vocab[1] = 'a' + 'b' = 'ab'.

# self.vocab[5] = 'ab'

# self.inverse_vocab['ab'] = 5

# For (5, 5): 6:

# p0 = 5, p1 = 5, new_id = 6.

# merged_token = self.vocab[5] + self.vocab[5] = 'ab' + 'ab' = 'abab'.

# self.vocab[6] = 'abab'

# self.inverse_vocab['abab'] = 6

# Final State after train:

# self.vocab = {0: 'a', 1: 'b', 2: 'c', 3: 'Ġ', 4: '<|endoftext|>', 5: 'ab', 6: 'abab'}

# self.inverse_vocab = {'a': 0, 'b': 1, 'c': 2, 'Ġ': 3, '<|endoftext|>': 4, 'ab': 5, 'abab': 6}

# self.bpe_merges = {(0, 1): 5, (5, 5): 6}

# Final token_ids after training (last value in loop) was [6, 2, 2]. (This is not stored in the tokenizer object, it's just the result of the training process).

# Encoding and Decoding with Trained Tokenizer:

# Let's encode the original corpus "ababcc" and decode it back.

# Encoding "ababcc":

# encoded_ids = tokenizer.encode("ababcc")
# print(encoded_ids) # Output: [6, 2, 2]
# content_copy
# download
# Use code with caution.
# Python

# The encode method effectively uses the learned bpe_merges to tokenize the input. For "ababcc", it will apply the merges and result in token IDs [6, 2, 2]. Token ID 6 corresponds to "abab", and token ID 2 corresponds to "c". So, "ababcc" is tokenized as ["abab", "c", "c"].

# Decoding [6, 2, 2]:

# decoded_text = tokenizer.decode([6, 2, 2])
# print(decoded_text) # Output: ababcc
# content_copy
# download
# Use code with caution.
# Python

# The decode method uses self.vocab to convert token IDs back to strings and concatenates them.
#endregion


class BPETokenizerSimple:
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """
        Train the BPE tokenizer from scratch.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """

        # Preprocess: Replace spaces with 'Ġ'
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including 'Ġ' if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]

        # Extend unique_chars with characters from processed_text that are not already included
        unique_chars.extend(char for char in sorted(
            set(processed_text)) if char not in unique_chars)

        # Optionally, ensure 'Ġ' is included if it is relevant to your text processing
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        # Now create the vocab and inverse vocab dictionaries
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:  # No more pairs to merge. Stopping training.
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.

        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe').
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # Convert loaded vocabulary to correct format
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        # Handle newline character without adding a new token
        if "\n" not in self.inverse_vocab:
            # Use an existing token ID as a placeholder for '\n'
            # Preferentially use "<|endoftext|>" if available
            fallback_token = next((token for token in [
                                  "<|endoftext|>", "Ġ", ""] if token in self.inverse_vocab), None)
            if fallback_token is not None:
                newline_token_id = self.inverse_vocab[fallback_token]
            else:
                # If no fallback token is available, raise an error
                raise KeyError(
                    "No suitable token found in vocabulary to map '\\n'.")

            self.inverse_vocab["\n"] = newline_token_id
            self.vocab[newline_token_id] = "\n"

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # Skip header line if present
            if lines and lines[0].startswith("#"):
                lines = lines[1:]

            for line in lines:
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    token1, token2 = pair
                    if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                        token_id1 = self.inverse_vocab[token1]
                        token_id2 = self.inverse_vocab[token2]
                        merged_token = token1 + token2
                        if merged_token in self.inverse_vocab:
                            merged_token_id = self.inverse_vocab[merged_token]
                            self.bpe_merges[(token_id1, token_id2)
                                            ] = merged_token_id
                        # print(f"Loaded merge: '{token1}' + '{token2}' -> '{merged_token}' (ID: {merged_token_id})")
                        else:
                            print(
                                f"Merged token '{merged_token}' not found in vocab. Skipping.")
                    else:
                        print(
                            f"Skipping pair {pair} as one of the tokens is not in the vocabulary.")

    def encode(self, text):
        """
        Encode the input text into a list of token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: The list of token IDs.
        """
        tokens = []
        # First split on newlines to preserve them
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")  # Add newline token separately
            words = line.split()
            for j, word in enumerate(words):
                if j == 0:
                    if i > 0:  # Start of a new line but not the first line
                        # Ensure it's marked as a new segment
                        tokens.append("Ġ" + word)
                    else:
                        tokens.append(word)
                else:
                    # Prefix words in the middle of a line with 'Ġ'
                    tokens.append("Ġ" + word)

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                # token is contained in the vocabulary as is
                token_ids.append(self.inverse_vocab[token])
            else:
                # Attempt to handle subword tokenization via BPE
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)

        return token_ids

    #region Purpose of tokenize_with_bpe(self, token) Function:
        # The main purpose of this function is to break a single token (word or part of a word) into smaller sub-tokens using BPE merge rules. This function is called within the encode function if a token (a word from the text) cannot be found directly in the vocabulary.

        # Step by Step Explanation Through Example:

        # Let's say we have sub-words "un", "der", "ful" in our vocabulary and we learned the combinations "un", "der", "ful" with our BPE merge rules. Let's say the word that is not in Vocabulary but needs to be tokenized is "underful".

        # Steps of tokenize_with_bpe("underful") call:

        # Character Level Initial Tokens:

        # token_ids = [self.inverse_vocab.get(char, None) for char in token]
        # # token = "underful"
        # # self.inverse_vocab = {'u': 10, 'n': 11, 'd': 12, 'e': 13, 'r': 14, 'f': 15, 'l': 16, ...} (example)
        # # token_ids = [10, 11, 12, 13, 14, 15, 16, 17] (character IDs u, n, d, e, r, f, u, l)
        # content_copy
        # download
        # Use code with caution.
        # Python

        # The word "underful" is split into characters: ['u', 'n', 'd', 'e', ​​'r', 'f', 'u', 'l'].

        # Each character has its ID in the vocabulary (assume the IDs are as above).

        # token_ids is initially a list of character IDs: [10, 11, 12, 13, 14, 15, 16, 17].

        # BPE Merge Cycle:

        # can_merge = True
        # while can_merge and len(token_ids) > 1:
        #  can_merge = False
        #  new_tokens = []
        #  i = 0
        #  while i < len(token_ids) - 1:
        #  pair = (token_ids[i], token_ids[i + 1])
        #  if pair in self.bpe_merges:
        #  merged_token_id = self.bpe_merges[pair]
        #  new_tokens.append(merged_token_id)
        #  i += 2
        #  can_merge = True
        #  else:
        #  new_tokens.append(token_ids[i])
        #  i += 1
        #  if i < len(token_ids):
        #  new_tokens.append(token_ids[i])
        #  token_ids = new_tokens
        # content_copy
        # download
        # Use code with caution.
        # Python

        # Pass 1:

        # token_ids = [10, 11, 12, 13, 14, 15, 16, 17]

        # First pair: pair = (token_ids[0], token_ids[1]) = (10, 11) (pair u, n).

        # Let's say the pair (10, 11) is not in self.bpe_merges.

        # Then token_ids[0] = 10 is added to new_tokens. new_tokens = [10]. i = 1.

        # Next pair: pair = (token_ids[1], token_ids[2]) = (11, 12) (pair n, d).

        # Let's say the pair (11, 12) is not in self.bpe_merges.

        # Token_ids[1] = 11 is added to new_tokens. new_tokens = [10, 11]. i = 2.

        # ... In this way, all pairs are checked and if no mergeable pair is found, new_tokens and token_ids remain the same.

        # Let's assume that no mergeable pair is found in this pass. new_tokens and token_ids do not change. can_merge = remains False. The loop ends.

        # Let's assume that in Pass 1, the pair (10, 11) (u, n) is in self.bpe_merges and the merge ID is 200.

        # First pair: pair = (token_ids[0], token_ids[1]) = (10, 11). The pair (10, 11) is in self.bpe_merges.

        # merged_token_id = self.bpe_merges[(10, 11)] = 200.

        # Add merged_token_id = 200 to new_tokens. new_tokens = [200]. i = 2 (skip next token). can_merge = True (merge done).

        # Continue looping for remaining token_ids = [12, 13, 14, 15, 16, 17].

        # Suppose at the end of Pass 1 the list of token_ids is [200, 201, 202, 17] (for example, "un", "der", "ful" merges were done and the last "l" could not be merged). can_merge = True.

        # Pass 2:

        # token_ids = [200, 201, 202, 17] (list from previous pass).

        # The loop starts again, this time with the token_ids list.

        # The pairs are checked again, if there is a pair that can be merged, it is merged, otherwise the pass ends.

        # Let's assume that (200, 201) could be merged in this pass and the token_ids list is [300, 202, 17]. can_merge = True.

        # Pass 3:

        # token_ids = [300, 202, 17].

        # The loop starts again. Let's assume that there are no more pairs that can be merged in this pass. can_merge = False. The loop ends.

        # Result:

        # The latest token_ids list (for example [300, 202, 17]) is returned by the function. This list represents the sub-token IDs obtained as a result of BPE tokenization of the word "underful".
    #endregion
    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(
                token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.bpe_merges:
                    merged_token_id = self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    # Uncomment for educational purposes:
                    # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                    i += 2  # Skip the next token as it's merged
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return token_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "  # Add space if not present before a newline
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced
