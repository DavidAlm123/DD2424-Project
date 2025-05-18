import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import re
import random
from collections import Counter
import tensorflow as tf
from keras import layers
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR =  BASE_DIR / "data" / "shaketext.txt"
fname = DATASET_DIR

with open(fname, "r") as fid:
    data = fid.read()

#data = data[:5000]

unique_chars = list(set(data))
K = len(unique_chars)
unique_chars_sorted = sorted(unique_chars)

char_to_index = {char: index for index, char in enumerate(unique_chars_sorted)}
index_to_char = {index: char for index, char in enumerate(unique_chars_sorted)}

# print("Total characters:", len(data))
# print("Unique characters (K):", K)
# print("Sample char to index mapping:", list(char_to_index.items())[:10])

text_as_int = [char_to_index[c] for c in data]

# Split the data: 70% train, 15% val, 15% test
total_len = len(text_as_int)
train_end = int(0.7 * total_len)
val_end = int(0.85 * total_len)

train_text = text_as_int[:train_end]
val_text = text_as_int[train_end:val_end]
test_text = text_as_int[val_end:]

seq_length = 100

char_train_dataset = tf.data.Dataset.from_tensor_slices(train_text)
char_val_dataset = tf.data.Dataset.from_tensor_slices(val_text)
char_test_dataset = tf.data.Dataset.from_tensor_slices(test_text)

sequences_train = char_train_dataset.batch(seq_length + 1, drop_remainder=True)
sequences_val = char_val_dataset.batch(seq_length + 1, drop_remainder=True)
sequences_test = char_test_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

train_dataset = sequences_train.map(split_input_target)
val_dataset = sequences_val.map(split_input_target)
test_dataset = sequences_test.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


for x_batch, y_batch in train_dataset.take(10):
    print("Input batch shape:", x_batch.shape)
    print("Target batch shape:", y_batch.shape)
    first_input = ''.join(index_to_char[idx] for idx in x_batch[0].numpy())
    first_target = ''.join(index_to_char[idx] for idx in y_batch[0].numpy())
    print("Decoded input:", first_input)
    print("Decoded target:", first_target)
    break

# Calculate steps_per_epoch for the learning rate schedule
# This needs to be after train_text, seq_length, and BATCH_SIZE are defined.
num_train_sequences = len(train_text) // (seq_length + 1)
steps_per_epoch = num_train_sequences // BATCH_SIZE

rnn_units = 100
embedding_dim = rnn_units//2

model = tf.keras.Sequential([
    layers.Embedding(input_dim=K, output_dim=embedding_dim),
    layers.SimpleRNN(rnn_units, return_sequences=True),
    layers.Dense(K)
])

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# model.compile(optimizer='adam', loss=loss_fn)

# def sample(model, start_string, generation_length=500, temperature=1.0):
#     input_eval = [char_to_index[s] for s in start_string]
#     input_eval = tf.expand_dims(input_eval, 0)
#     generated = []
#
#     for _ in range(generation_length):
#         predictions = model(input_eval)
#         predictions = tf.squeeze(predictions, 0) / temperature
#         predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
#         input_eval = tf.expand_dims([predicted_id], 0)
#         generated.append(index_to_char[predicted_id])
#
#     return start_string + ''.join(generated)
#
# print('Generated text pre-training:')
# print(sample(model, start_string=data[:6], generation_length=300))
# print()
#
# EPOCHS = 10
# #history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
#
# #model.save_weights('BASE_DIR/"Project_RNN".h5') # Corrected path if it were active
#
# print()
# print('Generated text post-training:')
# print(sample(model, start_string=data[:6], generation_length=300))


# from tqdm import tqdm # Already imported if LSTM section is active

# ... existing code ...
# rng.bit_generator.state = BitGen(seed).state

# def initialize_lstm(L, m):
#     layers = []
# ... existing code ...
#     return grads

# def train_lstm_adam_with_tracking(train_dataset, val_dataset, init_RNN, params):
#     import matplotlib.pyplot as plt # This import is fine even if section is commented
#
#     eta = params['eta']
# ... existing code ...
#     return RNN, history

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sample_text_lstm(RNN, start_string, generation_length, temperature=1.0):
#     layers = RNN['layers']
# ... existing code ...
#     return start_string + ''.join(generated)

# Initialize the first LSTM model (lstm1)
# init_lstm1 = initialize_lstm(1, 100) # L=1, m=100 (hidden units)

# print('Generated text pre-training (LSTM1):')
# print(sample_text_lstm(init_lstm1, data[:6], 300)) # Sample with initial random weights
# print()

# # Define hyperparameters for LSTM training
# # This 'params' dictionary will be used for both lstm1 and lstm2
# params = {
#     'eta': 0.001,                     # Learning rate
#     'num_epochs': 10,                 # Number of epochs for training
#     'beta1': 0.9,                     # Adam optimizer beta1
#     'beta2': 0.999,                   # Adam optimizer beta2
#     'eps': 1e-8,                      # Adam optimizer epsilon to prevent division by zero
#     'l2': 1e-4,                       # L2 regularization strength
#     'dropout': 0.2,                   # Dropout rate (used within train_lstm_adam_with_tracking)
#     'early_stopping_patience': 3,     # Patience for early stopping
#     'verbose': True                   # Whether to print progress during training
# }

# print("\nTraining LSTM model on original data (lstm1)...")
# # Train the first LSTM model using the original dataset
# lstm1, history1 = train_lstm_adam_with_tracking(
#     train_dataset,    # Original training data
#     val_dataset,      # Original validation data
#     init_lstm1,       # Initialized model weights and structure
#     params            # Hyperparameters
# )

# print('\nGenerated text post-training (LSTM1):')
# print(sample_text_lstm(lstm1, 'ROMEO.', 300)) # Sample with trained weights

# # Plot training history for lstm1
# plt.figure(figsize=(10, 4)) # Adjusted for potentially two plots if history2 is also plotted later

# plt.subplot(1, 2, 1) # Assuming you might plot lstm2 history in subplot 2 later
# plt.plot(history1['train_loss'], label='LSTM1 Train Loss')
# plt.plot(history1['val_loss'], label='LSTM1 Val Loss')
# plt.title("LSTM1 Loss over Epochs")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history1['train_acc'], label='LSTM1 Train Acc')
# plt.plot(history1['val_acc'], label='LSTM1 Val Acc')
# plt.title('LSTM1 Accuracy over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()


# ---------------------------
# 1. Data Augmentation (Character-level and Word-level EDA)
# ---------------------------
# def augment_text_char_level(text, noise_level=0.03): # Renamed from augment_text
#     """ Randomly swaps, drops, or duplicates characters to augment text """
#     chars = list(text)
#     n = len(chars)
#     new_chars = []
# 
#     for i in range(n):
#         if np.random.rand() < noise_level:
#             op = random.choice(['swap', 'drop', 'duplicate'])
#             if op == 'swap' and i < n - 1:
#                 chars[i], chars[i + 1] = chars[i + 1], chars[i]
#             elif op == 'drop':
#                 continue
#             elif op == 'duplicate':
#                 new_chars.append(chars[i])
# 
#         new_chars.append(chars[i])
#     return ''.join(new_chars)
# 
# # Word-level EDA style augmentations (inspired by the blog post)
# def eda_random_word_swap(text, n_swaps_ratio=0.05):
#     """ Randomly swaps words in the text. """
#     words = text.split(' ') # Split by space to preserve inter-word characters
#     actual_words_indices = [i for i, word in enumerate(words) if word.strip() != '']
#     
#     if len(actual_words_indices) < 2:
#         return text
#         
#     n_swaps = int(len(actual_words_indices) * n_swaps_ratio)
#     # Ensure at least one swap if possible for small texts with enough words
#     if n_swaps == 0 and len(actual_words_indices) >= 2:
#         n_swaps = 1
# 
#     for _ in range(n_swaps):
#         if len(actual_words_indices) < 2: # Break if list becomes too short
#             break
#         # Sample indices from the list of actual_words_indices
#         sampled_indices_in_actual_list = random.sample(range(len(actual_words_indices)), 2)
#         # Get the actual indices in the original 'words' list
#         idx1_in_words = actual_words_indices[sampled_indices_in_actual_list[0]]
#         idx2_in_words = actual_words_indices[sampled_indices_in_actual_list[1]]
#         
#         words[idx1_in_words], words[idx2_in_words] = words[idx2_in_words], words[idx1_in_words]
#     return " ".join(words)
# 
# def eda_random_word_deletion(text, p_delete=0.05):
#     """ Randomly deletes words from the text. """
#     words = text.split(' ') # Split by space
#     if not words:
#         return ""
#     
#     original_actual_word_count = sum(1 for word in words if word.strip() != '')
#     remaining_words = []
#     deleted_actual_word_count = 0
#     
#     for word in words:
#         is_actual_word = word.strip() != ''
#         if is_actual_word and random.random() < p_delete:
#             deleted_actual_word_count += 1
#             continue # Skip appending this word
#         remaining_words.append(word)
#         
#     # If all actual words were deleted, return the original text to avoid empty output
#     if original_actual_word_count > 0 and deleted_actual_word_count == original_actual_word_count:
#         return text
# 
#     return " ".join(remaining_words)
# 
# # Master augmentation function
# def apply_augmentations(original_text,
#                         char_noise_level=0.01,
#                         word_swap_ratio=0.02,
#                         word_delete_prob=0.02,
#                         enable_char_aug=True,
#                         enable_word_swap=True,
#                         enable_word_delete=True):
#     augmented_text = original_text
# 
#     if enable_char_aug:
#         augmented_text = augment_text_char_level(augmented_text, noise_level=char_noise_level)
#     
#     if enable_word_swap:
#         augmented_text = eda_random_word_swap(augmented_text, n_swaps_ratio=word_swap_ratio)
#     
#     if enable_word_delete:
#         augmented_text = eda_random_word_deletion(augmented_text, p_delete=word_delete_prob)
#         
#     return augmented_text
# 
# # Apply data augmentation to training data (before conversion to int)
# print("Applying data augmentations for the second LSTM model...")
# augmented_data = apply_augmentations(data, # Apply to the full 'data' string
#                                      char_noise_level=0.01,
#                                      word_swap_ratio=0.02, 
#                                      word_delete_prob=0.02,
#                                      enable_char_aug=True,
#                                      enable_word_swap=True,
#                                      enable_word_delete=True)
# print(f"Length of original data: {len(data)}")
# print(f"Length of augmented data: {len(augmented_data)}")
# if len(data) > 0 and len(augmented_data) < len(data) * 0.5 : # Check if data is not empty before division
#     print("Warning: Augmented data is significantly shorter than original. Check augmentation parameters.")


# # Ensure `char_to_index` is defined based on the original data's vocabulary
# # unique_chars = list(set(data)) # This should be from the *original* data to maintain vocab
# # unique_chars_sorted = sorted(unique_chars)
# # char_to_index = {char: index for index, char in enumerate(unique_chars_sorted)}
# # index_to_char = {index: char for char, index in char_to_index.items()} # Ensure this is also updated
# 
# # text_as_int_augmented = [char_to_index[c] for c in augmented_data if c in char_to_index]
# # # ^^^ IMPORTANT: Ensure only known characters are converted, or handle unknown characters.
# # # The existing script (lines 589-592) redefines char_to_index from original 'data',
# # # which is correct. If augmented_data contains new chars, it would fail.
# # # The current augmentations (char swap/drop/dup, word swap/del) don't create new char types.
# 
# # This part should follow the `augmented_data` creation:
# # (Ensure this uses the correct `char_to_index` from the original `data`)
# # unique_chars = list(set(data)) # Based on original data
# # unique_chars_sorted = sorted(unique_chars)
# # char_to_index = {char: index for index, char in enumerate(unique_chars_sorted)}
# # index_to_char = {index: char for char, index in char_to_index.items()} # Correctly re-derived from original
# # text_as_int_augmented = [char_to_index[c] for c in augmented_data]
# 
# # Then the splitting of augmented data:
# # total_len_aug = len(text_as_int_augmented) # Renamed to avoid conflict if original total_len is used elsewhere
# # train_end_aug = int(0.7 * total_len_aug)
# # val_end_aug = int(0.85 * total_len_aug)
# 
# # train_text_aug = text_as_int_augmented[:train_end_aug]
# # val_text_aug = text_as_int_augmented[train_end_aug:val_end_aug]
# # test_text_aug = text_as_int_augmented[val_end_aug:]
# 
# # Compare two LSTM models trained on original and augmented data
# # from tqdm import tqdm # Already imported
# 
# print("Training LSTM on augmented data (lstm2) - SKIPPED") # Modified print
# 
# # char_train_dataset_aug = tf.data.Dataset.from_tensor_slices(train_text_aug)
# # char_val_dataset_aug = tf.data.Dataset.from_tensor_slices(val_text_aug)
# # char_test_dataset_aug = tf.data.Dataset.from_tensor_slices(test_text_aug)
# 
# # def split_input_target(chunk): # This function is already defined earlier
# #     input_text = chunk[:-1]
# #     target_text = chunk[1:]
# #     return input_text, target_text
# 
# # train_dataset_aug = sequences_train_aug.map(split_input_target).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# # val_dataset_aug = sequences_val_aug.map(split_input_target).batch(BATCH_SIZE, drop_remainder=True)
# # test_dataset_aug = sequences_test_aug.map(split_input_target).batch(BATCH_SIZE, drop_remainder=True)
# 
# # init_lstm2 = initialize_lstm(1, 100)
# # lstm2, history2 = train_lstm_adam_with_tracking(train_dataset_aug, val_dataset_aug, init_lstm2, params)


# # Save
# # with open("lstm1_model.pkl", "wb") as f:
# #     pickle.dump(lstm1, f)
# 
# # with open("lstm2_model.pkl", "wb") as f:
# #     pickle.dump(lstm2, f)

# ---------------------------
# 3. Transformer Model (nano-GPT inspired, character-level)
# ---------------------------

def create_look_ahead_mask(size): # Reinstated function
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (size, size)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=rate)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu", use_bias=False), # Added use_bias=False
                layers.Dense(embed_dim, use_bias=False) # Added use_bias=False
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6) # nanoGPT often uses 1e-5, but 1e-6 is also common
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, look_ahead_mask, training=None):
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=look_ahead_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, dropout_rate=0.1): # Added dropout_rate
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # Learned positional embeddings
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.dropout = layers.Dropout(dropout_rate) # Added dropout layer

    def call(self, x, training=None): # Added training argument for dropout
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        
        embedded_tokens = self.token_emb(x)
        embedded_positions = self.pos_emb(positions)
        
        # Add token and position embeddings
        x = embedded_tokens + embedded_positions
        x = self.dropout(x, training=training) # Apply dropout
        return x

class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, maxlen, rate=0.1):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.maxlen = maxlen

        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, dropout_rate=rate) # Pass dropout_rate
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(rate) # Final dropout before output layer
        self.final_layer = layers.Dense(vocab_size)

    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1] # Needed for mask creation
        look_ahead_mask = create_look_ahead_mask(seq_len) # Create mask
        
        x = self.embedding_layer(inputs) # (batch_size, seq_len, embed_dim)
        
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, look_ahead_mask=look_ahead_mask, training=training) # Pass look_ahead_mask
            
        x = self.dropout(x, training=training)
        return self.final_layer(x) # (batch_size, seq_len, vocab_size)

# Transformer Hyperparameters (character-level)
TRANSFORMER_NUM_LAYERS = 2  # Increased from 2 to 4 for better capacity
TRANSFORMER_EMBED_DIM = 256 # Increased from 128
TRANSFORMER_NUM_HEADS = 4   # Number of attention heads
TRANSFORMER_FF_DIM = 4 * TRANSFORMER_EMBED_DIM    # Hidden layer size in FFN, increased
TRANSFORMER_MAXLEN = seq_length # Max sequence length for positional encoding, same as RNN/LSTM
TRANSFORMER_RATE = 0.1      # Dropout rate, can be 0.1 or 0.2. nanoGPT often uses 0.0 for small models, 0.1 or 0.2 for larger. Let's try 0.1
WEIGHT_DECAY = 0.01        # Weight decay for AdamW
LEARNING_RATE = 6e-4       # Initial learning rate (nanoGPT default for char-level)
WARMUP_EPOCHS = 1          # Number of epochs for warmup (e.g., 1-2 epochs)
MIN_LEARNING_RATE = 6e-5   # Minimum learning rate for cosine decay

warmup_steps_calculated = WARMUP_EPOCHS * steps_per_epoch

# Cosine Decay with Warmup Learning Rate Schedule - Keras Subclass
class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate_base,
                 total_steps,
                 warmup_steps_arg,
                 min_learning_rate=0.0,
                 warmup_learning_rate=0.0,
                 hold_base_rate_steps=0):
        super(CustomLearningRateSchedule, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = float(total_steps)
        self.warmup_steps_arg = float(warmup_steps_arg)
        self.min_learning_rate = float(min_learning_rate)
        self.warmup_learning_rate = float(warmup_learning_rate)
        self.hold_base_rate_steps = float(hold_base_rate_steps)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        if self.total_steps < self.warmup_steps_arg:
            raise ValueError('total_steps must be larger or equal to warmup_steps_arg.')

        learning_rate = 0.5 * self.learning_rate_base * (1 + tf.cos(
            np.pi * (step - self.warmup_steps_arg - self.hold_base_rate_steps) / 
            (self.total_steps - self.warmup_steps_arg - self.hold_base_rate_steps)))
        
        if self.hold_base_rate_steps > 0:
            learning_rate = tf.where(step > self.warmup_steps_arg + self.hold_base_rate_steps,
                                     learning_rate, self.learning_rate_base)
        
        if self.warmup_steps_arg > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    'learning_rate_base must be larger or equal to warmup_learning_rate.'
                )
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps_arg
            warmup_rate = slope * step + self.warmup_learning_rate
            learning_rate = tf.where(step < self.warmup_steps_arg, warmup_rate,
                                     learning_rate)
        
        final_lr = tf.maximum(learning_rate, self.min_learning_rate)
        return final_lr

    def get_config(self):
        return {
            "learning_rate_base": self.learning_rate_base,
            "total_steps": self.total_steps,
            "warmup_steps_arg": self.warmup_steps_arg,
            "min_learning_rate": self.min_learning_rate,
            "warmup_learning_rate": self.warmup_learning_rate,
            "hold_base_rate_steps": self.hold_base_rate_steps
        }

# Define TRANSFORMER_EPOCHS here, so it's clear for schedule creation
TRANSFORMER_EPOCHS = 100 # This should be the actual number of epochs for training

# Recalculate total_training_steps with the final TRANSFORMER_EPOCHS
total_training_steps_for_schedule = TRANSFORMER_EPOCHS * steps_per_epoch

# Instantiate the custom learning rate schedule class
lr_schedule_fn = CustomLearningRateSchedule(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_training_steps_for_schedule,
    warmup_steps_arg=warmup_steps_calculated,
    min_learning_rate=MIN_LEARNING_RATE
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule_fn, # Use the schedule
    weight_decay=WEIGHT_DECAY,
    beta_1=0.9, # Common Adam betas
    beta_2=0.95, # nanoGPT often uses 0.95 for beta2
    epsilon=1e-8,
    clipnorm=1.0 # Keep gradient clipping
)

transformer_model = TransformerModel(
    num_layers=TRANSFORMER_NUM_LAYERS,
    embed_dim=TRANSFORMER_EMBED_DIM,
    num_heads=TRANSFORMER_NUM_HEADS,
    ff_dim=TRANSFORMER_FF_DIM,
    vocab_size=K,
    maxlen=TRANSFORMER_MAXLEN,
    rate=TRANSFORMER_RATE
)

# Compile the Transformer model
transformer_model.compile(
    optimizer=optimizer, # Use the Adam optimizer with custom LR schedule
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Build the model by calling it on a sample batch from train_dataset
# This is to initialize weights and allow summary()
for input_example_batch, target_example_batch in train_dataset.take(1):
    transformer_model(input_example_batch, training=False)
    break
transformer_model.summary()

print("\nTraining Transformer model...")
# TRANSFORMER_EPOCHS = 50 # This is now defined above for the LR schedule
# To match the history object structure for later plotting if needed
history_transformer = transformer_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=TRANSFORMER_EPOCHS
)

# Reinserting the sample_transformer function definition
def sample_transformer(model, start_string, generation_length=500, temperature=1.0):
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0) # (1, num_chars_in_start_string)
    
    generated_chars = []

    for _ in range(generation_length):
        # The model expects (batch_size, seq_len)
        # During generation, seq_len can change.
        # Ensure input_eval does not exceed the model's maxlen for positional encoding
        if tf.shape(input_eval)[1] > model.maxlen:
            input_eval = input_eval[:, -model.maxlen:]

        predictions = model(input_eval, training=False) # (1, current_seq_len, vocab_size)
        
        # We only care about the prediction for the last token
        predictions = predictions[:, -1, :] / temperature # (1, vocab_size)
        
        # Apply categorical sampling
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0,0].numpy()
        
        # Add the predicted character to the input sequence for the next iteration
        input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=1) # (1, new_seq_len)
        
        generated_chars.append(index_to_char[predicted_id])

    return start_string + ''.join(generated_chars)

# Moved sampling block to after training
print("\nGenerated text from Transformer *after training*:")
print(sample_transformer(
        transformer_model,
        start_string="ROMEO:", # Changed start string
        generation_length=300,
        temperature=0.8)
)

# Added final loss printing
print(f"Final training loss:     {history_transformer.history['loss'][-1]:.3f}")
print(f"Final validation  loss:  {history_transformer.history['val_loss'][-1]:.3f}")

# Save the Transformer model weights
output_dir = BASE_DIR / "Project_RNN"
output_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
transformer_model.save_weights(str(output_dir / f"transformer_char_model_L{transformer_model.num_layers}_E{TRANSFORMER_EPOCHS}.weights.h5"))
print("\nTransformer model weights saved.")

# Optional: Plot Transformer training history (similar to other models)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_transformer.history['loss'], label='Train Loss')
plt.plot(history_transformer.history['val_loss'], label='Val Loss')
plt.title('Transformer Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_transformer.history['accuracy'], label='Train Acc')
plt.plot(history_transformer.history['val_accuracy'], label='Val Acc')
plt.title('Transformer Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

print("\nEvaluating Transformer model on test set...")
loss_transformer, acc_transformer = transformer_model.evaluate(test_dataset)
print(f"Transformer Test Loss: {loss_transformer:.4f}")
print(f"Transformer Test Accuracy: {acc_transformer:.4f}")

print("\nGenerating text with trained Transformer for qualitative evaluation (standard sampling):")
gen_text_transformer_standard = sample_transformer(transformer_model, "ROMEO:", 300, temperature=0.8)
print(gen_text_transformer_standard)

# --- Nucleus Sampling and Metrics Utilities (moved here or redefined for clarity) ---
def nucleus_sample(probs, p=0.9):
    """ Given a probability distribution, sample using top-p (nucleus) strategy """
    probs = np.asarray(probs).astype("float64") # Ensure float64 for precision
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, p)
    # Ensure cutoff is at least 0 and includes at least one item if p > 0
    chosen_indices = sorted_indices[:cutoff + 1]
    if not len(chosen_indices) and p > 0 and len(probs) > 0:
         chosen_indices = sorted_indices[:1] # Fallback to top-1 if nucleus is empty
    
    if not len(chosen_indices): # If still empty (e.g. all probs are zero or p=0)
        # Fallback: sample from all available indices if possible, or raise error/return default
        # For now, let's pick a random one if vocab is not empty, or the first one.
        if len(probs) > 0:
            # This case should be rare if probs is a valid distribution
            # return np.random.choice(np.arange(len(probs)))
            # More robust: pick the most probable if all else fails and chosen_indices is empty.
             return np.argmax(probs) if len(probs) > 0 else 0
        else: # Should not happen with a K > 0 vocabulary
            return 0 

    chosen_probs = probs[chosen_indices]
    chosen_probs /= chosen_probs.sum() # Normalize the chosen probabilities
    return np.random.choice(chosen_indices, p=chosen_probs)

def sample_transformer_nucleus(model, start_string, generation_length=500, temperature=1.0, top_p=0.9):
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    generated_chars = []

    for _ in range(generation_length):
        if tf.shape(input_eval)[1] > model.maxlen:
            input_eval = input_eval[:, -model.maxlen:]

        predictions = model(input_eval, training=False)
        predictions = predictions[:, -1, :] / temperature
        
        # Convert logits to probabilities for nucleus sampling
        probs = tf.nn.softmax(predictions, axis=-1).numpy().flatten()
        
        predicted_id = nucleus_sample(probs, p=top_p)
        
        input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=1)
        generated_chars.append(index_to_char[predicted_id])

    return start_string + ''.join(generated_chars)

# --- Metrics Functions (moved here or redefined for clarity) ---
def compute_spelling_accuracy(text, word_list):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    correct = sum(w in word_list for w in words)
    return correct / len(words)

def ngram_overlap(generated_text, reference_text, n=2):
    def get_ngrams(text, n_gram):
        # Ensure text is long enough for n-gram extraction
        return set([text[i:i+n_gram] for i in range(len(text)-n_gram+1)])
    gen_ngrams = get_ngrams(generated_text, n)
    ref_ngrams = get_ngrams(reference_text, n)
    if not gen_ngrams: # Avoid division by zero if generated text is too short for n-grams
        return 0.0
    return len(gen_ngrams & ref_ngrams) / len(gen_ngrams)

def ngram_diversity(text, n=2):
    # Ensure text is long enough for n-gram extraction
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not ngrams: # Avoid division by zero if text is too short for n-grams
        return 0.0
    return len(set(ngrams)) / len(ngrams)
# --- End of Metrics Functions ---

print("\nGenerating text with trained Transformer (Nucleus Sampling):")
gen_text_transformer_nucleus = sample_transformer_nucleus(transformer_model, "ROMEO:", 300, temperature=0.8, top_p=0.9)
print(gen_text_transformer_nucleus)

# Quantitative Evaluation for Transformer with Nucleus Sampling
# reference_text is the full 'data'. word_list is derived from full 'data'.
# This is a general evaluation against the source material.
reference_text_for_metrics = data # Using full data as reference as per original commented code
word_list_for_metrics = set(re.findall(r'\b\w+\b', reference_text_for_metrics.lower()))

print("\nQuantitative Evaluation for Transformer (Nucleus Sampled Text):")
spelling_acc_transformer = compute_spelling_accuracy(gen_text_transformer_nucleus, word_list_for_metrics)
bigram_ov_transformer = ngram_overlap(gen_text_transformer_nucleus, reference_text_for_metrics, n=2)
trigram_div_transformer = ngram_diversity(gen_text_transformer_nucleus, n=3)
print(f"Transformer Model (Nucleus) -> Spelling Accuracy: {spelling_acc_transformer:.2f}, Bigram Overlap: {bigram_ov_transformer:.2f}, Trigram Diversity: {trigram_div_transformer:.2f}")


# Continue with existing Nucleus sampling and other evaluations for LSTM models if needed,
# or integrate Transformer into those comparisons.
# The following sections for Nucleus Sampling, original LSTM model evaluations, and word-level models are retained.

''' # Re-add start of block comment for the GLoVe/BPE part
# ---------------------------
# 2. Nucleus Sampling (Top-p Sampling)
# ---------------------------
def nucleus_sample(probs, p=0.9):
    """ Given a probability distribution, sample using top-p (nucleus) strategy """
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, p)
    chosen_indices = sorted_indices[:cutoff + 1]
    chosen_probs = probs[chosen_indices]
    chosen_probs /= chosen_probs.sum()
    return np.random.choice(chosen_indices, p=chosen_probs)

def sample_text_lstm_nucleus(RNN, start_string, generation_length, temperature=1.0, top_p=0.9):
    layers = RNN['layers']
    V, c_out = RNN['V'], RNN['c']
    L = len(layers)

    h_states = [np.zeros((m,1)) for _ in range(L)]
    c_states = [np.zeros((m,1)) for _ in range(L)]

    last_char_id = None
    for ch in start_string:
        x = np.zeros((K,1))
        last_char_id = char_to_index[ch]
        x[last_char_id,0] = 1
        for l in range(L):
            Wf, Uf, bf = layers[l]['Wf'], layers[l]['Uf'], layers[l]['bf']
            Wi, Ui, bi = layers[l]['Wi'], layers[l]['Ui'], layers[l]['bi']
            Wc, Uc, bc = layers[l]['Wc'], layers[l]['Uc'], layers[l]['bc']
            Wo, Uo, bo = layers[l]['Wo'], layers[l]['Uo'], layers[l]['bo']

            h_prev = h_states[l]
            c_prev = c_states[l]

            f_gate   = 1 / (1 + np.exp(-(Wf @ h_prev + Uf @ x + bf)))
            i_gate   = 1 / (1 + np.exp(-(Wi @ h_prev + Ui @ x + bi)))
            c_tilde  = np.tanh(Wc @ h_prev + Uc @ x + bc)
            c_curr   = f_gate * c_prev + i_gate * c_tilde
            o_gate   = 1 / (1 + np.exp(-(Wo @ h_prev + Uo @ x + bo)))
            h_curr   = o_gate * np.tanh(c_curr)

            h_states[l] = h_curr
            c_states[l] = c_curr
            x = h_curr

    generated = []
    for _ in range(generation_length):
        x = np.zeros((K,1))
        x[last_char_id,0] = 1

        for l in range(L):
            Wf, Uf, bf = layers[l]['Wf'], layers[l]['Uf'], layers[l]['bf']
            Wi, Ui, bi = layers[l]['Wi'], layers[l]['Ui'], layers[l]['bi']
            Wc, Uc, bc = layers[l]['Wc'], layers[l]['Uc'], layers[l]['bc']
            Wo, Uo, bo = layers[l]['Wo'], layers[l]['Uo'], layers[l]['bo']

            h_prev = h_states[l]
            c_prev = c_states[l]

            f_gate   = 1 / (1 + np.exp(-(Wf @ h_prev + Uf @ x + bf)))
            i_gate   = 1 / (1 + np.exp(-(Wi @ h_prev + Ui @ x + bi)))
            c_tilde  = np.tanh(Wc @ h_prev + Uc @ x + bc)
            c_curr   = f_gate * c_prev + i_gate * c_tilde
            o_gate   = 1 / (1 + np.exp(-(Wo @ h_prev + Uo @ x + bo)))
            h_curr   = o_gate * np.tanh(c_curr)

            h_states[l] = h_curr
            c_states[l] = c_curr
            x = h_curr

        logits = (V @ h_curr + c_out).flatten() / temperature
        exp_logits = np.exp(logits - np.max(logits))
        p = exp_logits / exp_logits.sum()
        last_char_id = nucleus_sample(p, p=top_p)
        generated.append(index_to_char[last_char_id])

    return start_string + ''.join(generated)

def compute_spelling_accuracy(text, word_list):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    correct = sum(w in word_list for w in words)
    return correct / len(words)

def ngram_overlap(generated_text, reference_text, n=2):
    def get_ngrams(text, n_gram):
        # Ensure text is long enough for n-gram extraction
        return set([text[i:i+n_gram] for i in range(len(text)-n_gram+1)])
    gen_ngrams = get_ngrams(generated_text, n)
    ref_ngrams = get_ngrams(reference_text, n)
    if not gen_ngrams: # Avoid division by zero if generated text is too short for n-grams
        return 0.0
    return len(gen_ngrams & ref_ngrams) / len(gen_ngrams)

def ngram_diversity(text, n=2):
    # Ensure text is long enough for n-gram extraction
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not ngrams: # Avoid division by zero if text is too short for n-grams
        return 0.0
    return len(set(ngrams)) / len(ngrams)

# Qualitative comparison using nucleus sampling
print("\nGenerated text from LSTM trained on original data (nucleus sampling):")
print(sample_text_lstm_nucleus(lstm1, 'ROMEO.', 300))

print("\nGenerated text from LSTM trained on augmented data (nucleus sampling):")
print(sample_text_lstm_nucleus(lstm2, 'ROMEO.', 300))

# Quantitative comparison
reference_text = data  # Original clean text
word_list = set(re.findall(r'\b\w+\b', reference_text.lower()))

print("Quantitative Evaluation on Test Set:")
for name, model, test_dataset in [('Original', lstm1, test_dataset), ('Augmented', lstm2, test_dataset_aug)]:
    total_acc = 0
    total_ngrams = Counter()
    total_count = 0
    for x_batch, y_batch in test_dataset:
        for x_seq, y_seq in zip(x_batch.numpy(), y_batch.numpy()):
            pred_seq = []
            h0s = [np.zeros((m, 1)) for _ in range(1)]
            c0s = [np.zeros((m, 1)) for _ in range(1)]
            X = np.zeros((K, seq_length))
            Y = np.zeros((K, seq_length))
            for t in range(seq_length):
                X[x_seq[t], t] = 1
                Y[y_seq[t], t] = 1
            cache, _ = fp_lstm(model, X, Y, h0s, c0s)
            preds = np.stack(cache['p'], axis=1)
            preds_idx = np.argmax(preds, axis=0)
            true_idx = np.argmax(Y, axis=0)
            total_acc += np.sum(preds_idx == true_idx)
            total_count += seq_length
            preds_idx = preds_idx.flatten()  # Make it 1D
            pred_text = ''.join(index_to_char[idx] for idx in preds_idx)
            total_ngrams.update([pred_text[i:i+2] for i in range(len(pred_text)-1)])
    acc = total_acc / total_count
    unique_ngrams = len(total_ngrams)
    total_ngrams_count = sum(total_ngrams.values())
    diversity = unique_ngrams / total_ngrams_count if total_ngrams_count > 0 else 0
    print(f"{name} Model -> Test Accuracy: {acc:.2f}, Bigram Diversity: {diversity:.2f}")
for name, model in [('Original', lstm1), ('Augmented', lstm2)]:
    gen = sample_text_lstm_nucleus(model, 'ROMEO.', 300)
    acc = compute_spelling_accuracy(gen, word_list)
    bigram_ov = ngram_overlap(gen, reference_text, n=2)
    trigram_div = ngram_diversity(gen, n=3)
    print(f"{name} Model -> Spelling Accuracy: {acc:.2f}, Bigram Overlap: {bigram_ov:.2f}, Trigram Diversity: {trigram_div:.2f}")

''' # Re-add end of block comment



