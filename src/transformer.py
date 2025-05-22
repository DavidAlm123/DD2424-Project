import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import tensorflow as tf
from keras import layers

train = False  # Set to False to load a pre-trained model instead of training

BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
DATASET_DIR = BASE_DIR / "data" / "shaketext.txt"
fname = DATASET_DIR

with open(fname, "r") as fid:
    data = fid.read()

unique_chars = list(set(data))
K = len(unique_chars)
unique_chars_sorted = sorted(unique_chars)

char_to_index = {char: index for index, char in enumerate(unique_chars_sorted)}
index_to_char = {index: char for index, char in enumerate(unique_chars_sorted)}

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

num_train_sequences = len(train_text) // (seq_length + 1)
steps_per_epoch = num_train_sequences // BATCH_SIZE

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=rate)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu", use_bias=False),
                layers.Dense(embed_dim, use_bias=False)
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
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
    def __init__(self, maxlen, vocab_size, embed_dim, dropout_rate=0.1):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        
        embedded_tokens = self.token_emb(x)
        embedded_positions = self.pos_emb(positions)
        
        x = embedded_tokens + embedded_positions
        x = self.dropout(x, training=training)
        return x

class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, maxlen, rate=0.1):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.maxlen = maxlen

        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, dropout_rate=rate)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(rate)
        self.final_layer = layers.Dense(vocab_size)

    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1]
        look_ahead_mask = create_look_ahead_mask(seq_len)
        
        x = self.embedding_layer(inputs, training=training)
        
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, look_ahead_mask=look_ahead_mask, training=training)
            
        x = self.dropout(x, training=training)
        return self.final_layer(x)

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

# HYPERPARAMETERS
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_EMBED_DIM = 256
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_FF_DIM = 4 * TRANSFORMER_EMBED_DIM
TRANSFORMER_MAXLEN = seq_length
TRANSFORMER_RATE = 0.2
WEIGHT_DECAY = 0.01
LEARNING_RATE = 6e-4
WARMUP_EPOCHS = 1
MIN_LEARNING_RATE = 6e-5
TRANSFORMER_EPOCHS = 50 
EPOCHS_FOR_LOADING = 100

warmup_steps_calculated = WARMUP_EPOCHS * steps_per_epoch
total_training_steps_for_schedule = TRANSFORMER_EPOCHS * steps_per_epoch

def sample_transformer(model, start_string, generation_length=500):
    input_eval = [char_to_index[s] for s in start_string if s in char_to_index]
    if not input_eval:
        print("Warning: Start string became empty after filtering. Using a default character.")
        input_eval = [char_to_index[unique_chars_sorted[0]]] if unique_chars_sorted else [0]

    input_eval = tf.expand_dims(input_eval, 0)
    
    generated_chars = []

    for _ in range(generation_length):
        current_seq_len = tf.shape(input_eval)[1]
        if current_seq_len > model.maxlen:
            input_eval = input_eval[:, -model.maxlen:]

        predictions = model(input_eval, training=False)
        predictions = predictions[:, -1, :]
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0,0].numpy()
        input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=1)
        generated_chars.append(index_to_char[predicted_id])

    return start_string + ''.join(generated_chars)

def nucleus_sample(probs, p=0.9):
    probs = np.asarray(probs).astype("float64")
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, p)
    chosen_indices = sorted_indices[:cutoff + 1]
    
    if not chosen_indices.size and p > 0 and probs.size > 0:
         chosen_indices = sorted_indices[:1]
    
    if not chosen_indices.size:
        if probs.size > 0:
             return np.argmax(probs) if probs.size > 0 else 0
        else:
            return 0

    chosen_probs = probs[chosen_indices]
    chosen_probs /= chosen_probs.sum()
    return np.random.choice(chosen_indices, p=chosen_probs)

def sample_transformer_nucleus(model, start_string, generation_length=500, top_p=0.9):
    input_eval = [char_to_index[s] for s in start_string if s in char_to_index]
    if not input_eval:
        print("Warning: Start string became empty after filtering. Using a default character.")
        input_eval = [char_to_index[unique_chars_sorted[0]]] if unique_chars_sorted else [0]
        
    input_eval = tf.expand_dims(input_eval, 0)
    generated_chars = []

    for _ in range(generation_length):
        current_seq_len = tf.shape(input_eval)[1]
        if current_seq_len > model.maxlen:
            input_eval = input_eval[:, -model.maxlen:]

        predictions = model(input_eval, training=False)
        predictions = predictions[:, -1, :]
        
        probs = tf.nn.softmax(predictions, axis=-1).numpy().flatten()
        
        predicted_id = nucleus_sample(probs, p=top_p)
        
        input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=1)
        generated_chars.append(index_to_char[predicted_id])

    return start_string + ''.join(generated_chars)

def compute_spelling_accuracy(text, word_list):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    correct = sum(w in word_list for w in words)
    return correct / len(words)

def ngram_overlap(generated_text, reference_text, n=2):
    def get_ngrams(text, n_gram):
        if not isinstance(text, str) or len(text) < n_gram:
            return set()
        return set([text[i:i+n_gram] for i in range(len(text)-n_gram+1)])
    gen_ngrams = get_ngrams(generated_text, n)
    ref_ngrams = get_ngrams(reference_text, n)
    if not gen_ngrams:
        return 0.0
    return len(gen_ngrams & ref_ngrams) / len(gen_ngrams) if gen_ngrams else 0.0

def ngram_diversity(text, n=2):
    if not isinstance(text, str) or len(text) < n:
        return 0.0
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

lr_schedule_fn = CustomLearningRateSchedule(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_training_steps_for_schedule,
    warmup_steps_arg=warmup_steps_calculated,
    min_learning_rate=MIN_LEARNING_RATE
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule_fn,
    weight_decay=WEIGHT_DECAY,
    beta_1=0.9,
    beta_2=0.95,
    epsilon=1e-8,
    clipnorm=1.0
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

transformer_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

for input_example_batch, target_example_batch in train_dataset.take(1):
    transformer_model(input_example_batch, training=False)
    break
transformer_model.summary()

history_transformer = None
WEIGHTS_DIR = BASE_DIR / "data"  # Directory for saving/loading model weights

if train:
    print("\nTraining Transformer model...")
    history_transformer = transformer_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=TRANSFORMER_EPOCHS
    )
    
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    transformer_model.save_weights(str(WEIGHTS_DIR / f"transformer_char_model_L{transformer_model.num_layers}_E{TRANSFORMER_EPOCHS}.weights.h5"))
    print("\nTransformer model weights saved.")
    
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
    
else:
    model_weights_filename = f"transformer_char_model_L{TRANSFORMER_NUM_LAYERS}_E{EPOCHS_FOR_LOADING}.weights.h5"
    model_weights_path = WEIGHTS_DIR / model_weights_filename
    print(f"Attempting to load model from: {model_weights_path}")
    
    if model_weights_path.exists():
        try:
            transformer_model.load_weights(str(model_weights_path))
            print(f"Successfully loaded weights from: {model_weights_path}")
        except Exception as e:
            print(f"ERROR loading weights: {e}")
            print("Proceeding with an uninitialized model.")
    else:
        print(f"ERROR: Weights file not found at: {model_weights_path}")
        print("Proceeding with an uninitialized model.")

print("\nGenerating text with transformer model (standard sampling):")
generated_text_standard = sample_transformer(
    transformer_model,
    start_string="ROMEO:",
    generation_length=300
)
print(generated_text_standard)

print("\nGenerating text with transformer model (nucleus sampling, p=0.9):")
generated_text_nucleus = sample_transformer_nucleus(
    transformer_model,
    start_string="ROMEO:",
    generation_length=300,
    top_p=0.9
)
print(generated_text_nucleus)


reference_text_for_metrics = data
word_list_for_metrics = set(re.findall(r'\b\w+\b', reference_text_for_metrics.lower()))

print("\nQuantitative Text Evaluation:")

spelling_acc_std = compute_spelling_accuracy(generated_text_standard, word_list_for_metrics)
bigram_ov_std = ngram_overlap(generated_text_standard, reference_text_for_metrics, n=2)
trigram_div_std = ngram_diversity(generated_text_standard, n=3)
print(f"Standard Sampling -> Spelling Accuracy: {spelling_acc_std:.2f}, Bigram Overlap: {bigram_ov_std:.2f}, Trigram Diversity: {trigram_div_std:.2f}")

spelling_acc_nuc = compute_spelling_accuracy(generated_text_nucleus, word_list_for_metrics)
bigram_ov_nuc = ngram_overlap(generated_text_nucleus, reference_text_for_metrics, n=2)
trigram_div_nuc = ngram_diversity(generated_text_nucleus, n=3)
print(f"Nucleus Sampling -> Spelling Accuracy: {spelling_acc_nuc:.2f}, Bigram Overlap: {bigram_ov_nuc:.2f}, Trigram Diversity: {trigram_div_nuc:.2f}") 