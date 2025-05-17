import tensorflow as tf
import numpy as np
from transformer import Transformer, create_masks
import time
import re
import sys

def load_data(fname):
    with open(fname, "r") as fid:
        data = fid.read()
    return data

def create_vocab(data):
    unique_chars = sorted(list(set(data)))
    char_to_index = {char: index for index, char in enumerate(unique_chars)}
    index_to_char = {index: char for index, char in enumerate(unique_chars)}
    return char_to_index, index_to_char, len(unique_chars)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def train_transformer():
    # Hyperparameters (user confirmed these for the bad run, but may need more epochs)
    num_layers = 6
    d_model = 384
    num_heads = 6
    dff = 1536
    dropout_rate = 0.2
    seq_length = 100  
    batch_size = 64   
    epochs = 20 # SIGNIFICANTLY INCREASED EPOCHS
    buffer_size = 10000 # For shuffling
    
    # Load and preprocess data
    data_text = load_data('data/shaketext.txt')
    char_to_index, index_to_char, vocab_size = create_vocab(data_text)
    
    text_as_int = [char_to_index[c] for c in data_text]

    # Split the data: 70% train, 15% val, 15% test (as in test.py)
    total_len = len(text_as_int)
    train_end_idx = int(0.7 * total_len)
    val_end_idx = int(0.85 * total_len)

    train_text_as_int = text_as_int[:train_end_idx]
    val_text_as_int = text_as_int[train_end_idx:val_end_idx]
    test_text_as_int = text_as_int[val_end_idx:] # Test set created but not used in this training loop yet

    # Create char datasets
    char_train_dataset = tf.data.Dataset.from_tensor_slices(train_text_as_int)
    char_val_dataset = tf.data.Dataset.from_tensor_slices(val_text_as_int)
    # char_test_dataset = tf.data.Dataset.from_tensor_slices(test_text_as_int) # For later use

    # Create sequences
    sequences_train = char_train_dataset.batch(seq_length + 1, drop_remainder=True)
    sequences_val = char_val_dataset.batch(seq_length + 1, drop_remainder=True)
    # sequences_test = char_test_dataset.batch(seq_length + 1, drop_remainder=True) # For later use

    # Apply split_input_target
    dataset_train_processed = sequences_train.map(split_input_target)
    dataset_val_processed = sequences_val.map(split_input_target)
    # dataset_test_processed = sequences_test.map(split_input_target) # For later use

    # Shuffle and batch the final datasets
    train_dataset = dataset_train_processed.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    val_dataset = dataset_val_processed.batch(batch_size, drop_remainder=True) # No shuffle for val
    # test_dataset = dataset_test_processed.batch(batch_size, drop_remainder=True) # No shuffle for test, for later use
    
    total_train_batches_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    if total_train_batches_per_epoch == tf.data.experimental.UNKNOWN_CARDINALITY:
        print("Warning: Could not determine cardinality of training dataset. Logging of batch progress might be affected.")
        total_train_batches_per_epoch = -1 # Placeholder if unknown

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size,
        maximum_position_encoding=seq_length,
        rate=dropout_rate
    )
    transformer.build(input_shape=(None, seq_length)) # Input to transformer is seq_length, not seq_length-1

    learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.001
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    @tf.function
    def train_step(inp, tar):
        # tar_inp is inp, tar_real is tar, already split by split_input_target
        # The input to the transformer is `inp` which has seq_length.
        # The target for loss calculation is `tar` which also has seq_length.
        # The transformer itself will predict for each position based on previous ones.
        # Loss calculation should be: loss_fn(tar, predictions_for_tar)
        # where predictions_for_tar are the logits corresponding to *each token in tar*.

        # The model predicts the next token for each token in the input sequence.
        # So, if input is x_1, x_2, ..., x_T, it outputs p_1, p_2, ..., p_T
        # where p_i is prediction for y_i (which is x_{i+1})
        # So, we need to compare predictions for x_1...x_{T-1} with targets y_1...y_{T-1} (which are x_2...x_T)
        # The `split_input_target` gives: inp = chunk[:-1], tar = chunk[1:]
        # So `inp` is x_1...x_{T-1} and `tar` is x_2...x_T. Both have length `seq_length`.
        # This seems correct for typical seq-to-seq language modeling where input and target are shifted.
        # Let's call them `input_sequence` and `target_sequence` for clarity in this function.
        input_sequence = inp
        target_sequence = tar 

        combined_mask = create_masks(input_sequence) # Mask is for the input to the decoder-like transformer
        
        with tf.GradientTape() as tape:
            # The transformer takes `input_sequence` and its output logits should align with `target_sequence`
            predictions = transformer(inputs=input_sequence, training=True, mask=combined_mask) 
            loss = loss_fn(target_sequence, predictions)
        
        gradients = tape.gradient(loss, transformer.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(clipped_gradients, transformer.trainable_variables))
        return loss

    @tf.function
    def evaluate_step(inp, tar):
        input_sequence = inp
        target_sequence = tar
        combined_mask = create_masks(input_sequence)
        
        predictions = transformer(inputs=input_sequence, training=False, mask=combined_mask)
        loss = loss_fn(target_sequence, predictions)
        
        predicted_ids = tf.argmax(predictions, axis=-1, output_type=target_sequence.dtype)
        correct_predictions = tf.equal(predicted_ids, target_sequence)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return loss, accuracy

    # Training loop (largely the same, just uses new dataset names)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()
        total_train_loss = 0
        num_train_batches_processed = 0
        
        for batch_idx, (inp_batch, tar_batch) in enumerate(train_dataset):
            batch_start_time = time.time()
            loss = train_step(inp_batch, tar_batch)
            total_train_loss += loss
            num_train_batches_processed += 1
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            # Calculate batches_per_sec here for all cases
            batches_per_sec = 1.0 / batch_duration if batch_duration > 0 else float('inf') 
            
            if total_train_batches_per_epoch > 0 and batch_idx % 20 == 0:
                print(f"  Train Batch {batch_idx+1}/{total_train_batches_per_epoch}, Loss: {loss:.4f}, Speed: {batches_per_sec:.2f} batches/sec")
            elif total_train_batches_per_epoch == -1 and batch_idx % 20 == 0: # Fallback if cardinality is unknown
                 print(f"  Train Batch {batch_idx+1}, Loss: {loss:.4f}, Speed: {batches_per_sec:.2f} batches/sec")

        avg_train_loss = total_train_loss / num_train_batches_processed if num_train_batches_processed > 0 else 0
        epoch_train_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} Training - Avg Loss: {avg_train_loss:.4f}, Time: {epoch_train_duration:.2f} sec")

        val_epoch_start_time = time.time()
        total_val_loss = 0
        total_val_accuracy = 0
        num_val_batches_processed = 0
        for inp_val_batch, tar_val_batch in val_dataset:
            val_loss, val_accuracy = evaluate_step(inp_val_batch, tar_val_batch)
            total_val_loss += val_loss
            total_val_accuracy += val_accuracy
            num_val_batches_processed += 1
        
        if num_val_batches_processed > 0:
            avg_val_loss = total_val_loss / num_val_batches_processed
            avg_val_accuracy = total_val_accuracy / num_val_batches_processed
            val_epoch_duration = time.time() - val_epoch_start_time
            print(f"Epoch {epoch + 1} Validation - Avg Loss: {avg_val_loss:.4f}, Avg Accuracy: {avg_val_accuracy:.4f}, Time: {val_epoch_duration:.2f} sec")
        else:
            print(f"Epoch {epoch + 1} Validation - No validation batches processed.")

    # test_text_as_int is already defined from the 70/15/15 split.
    # We need it as string for metrics.
    test_text_str = ''.join([index_to_char[i] for i in test_text_as_int])

    transformer.save_weights('data/transformer.weights.h5')
    # Return the test_text_str along with other items so it can be used in __main__
    return transformer, char_to_index, index_to_char, test_text_str

def generate_text(transformer, start_string, char_to_index, index_to_char, generation_length=1000, temperature=1.0, top_p=0.9):
    # `input_eval` must be a tensor of shape (batch_size, seq_length)
    # `temperature` controls randomness: lower means more deterministic, higher means more surprising
    # `top_p` controls nucleus sampling: 0 < top_p < 1.0. If 0 or >=1, it's disabled.

    # Ensure temperature is not zero
    temperature = tf.maximum(temperature, 1e-6) # Prevent division by zero

    # char_to_ind_len = len(char_to_index) # This variable is not used further
    input_char_ids = [char_to_index[s] for s in start_string]
    input_eval = tf.constant([input_char_ids], dtype=tf.int32) # Explicitly tf.int32

    text_generated = []

    # transformer.reset_states() # This was for RNNs, remove for Transformer architecture

    for _ in range(generation_length):
        # `input_eval` shape: (batch_size, current_sequence_length)
        # Pad or truncate `input_eval` to `seq_length`
        current_seq_len = tf.shape(input_eval)[1]
        if current_seq_len < transformer.seq_length:
            # padding_size = transformer.seq_length - current_seq_len # Not directly used
            # Let's ensure input_eval is always seq_length by taking the last seq_length tokens
            # This slicing should happen *before* deciding to pad.
            # The input to the model should be the last `seq_length` tokens. 
            # If input_eval is shorter than seq_length, we pad it on the left.
            
            # Take the last part of input_eval, up to seq_length. If it's shorter, this is just input_eval.
            input_sequence_for_model_prep = input_eval[:, -transformer.seq_length:]
            current_prep_len = tf.shape(input_sequence_for_model_prep)[1]
            
            if current_prep_len < transformer.seq_length:
                 padding_amount = transformer.seq_length - current_prep_len
                 padding = tf.zeros((tf.shape(input_sequence_for_model_prep)[0], padding_amount), dtype=tf.int32) # Use tf.int32 for padding
                 input_for_model = tf.concat([padding, input_sequence_for_model_prep], axis=1)
            else:
                 input_for_model = input_sequence_for_model_prep

        elif current_seq_len > transformer.seq_length:
            input_for_model = input_eval[:, -transformer.seq_length:] # Take the last seq_length tokens
        else: # current_seq_len == transformer.seq_length
            input_for_model = input_eval

        # The model expects an input of shape (batch_size, seq_length).
        # tf.print(f"input_for_model shape: {tf.shape(input_for_model)}, dtype: {input_for_model.dtype}")


        predictions = transformer(input_for_model, training=False, mask=None)
        # `predictions` shape: (batch_size, seq_length, vocab_size)
        
        # We only need the predictions for the last token in the sequence
        predictions = predictions[:, -1, :]  # Shape: (batch_size, vocab_size)

        # Apply temperature
        predictions = predictions / temperature
        
        # Check for NaNs/Infs after temperature scaling
        tf.debugging.check_numerics(predictions, "Predictions after temperature scaling")


        # Nucleus Samping (Top-p filtering)
        if top_p > 0 and top_p < 1.0:
            # `predictions` are logits of shape (batch_size, vocab_size)
            
            # 1. Sort logits and get original indices
            sorted_logits, sorted_indices = tf.sort(predictions, direction='DESCENDING', axis=-1), tf.argsort(predictions, direction='DESCENDING', axis=-1)

            # 2. Calculate cumulative probabilities on sorted logits
            cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

            # 3. Create mask for elements to remove in the sorted list
            sorted_indices_to_remove_mask = cumulative_probs > top_p 

            # 4. Ensure at least one token is kept (the highest probability one) for each batch item.
            # This sets the first column (0th element in sorted list) of sorted_indices_to_remove_mask to False.
            # The tf.pad approach below is graph-compatible.
            paddings = [[0, 0], [1, 0]] 
            mask_with_false_prepended = tf.pad(sorted_indices_to_remove_mask, paddings, "CONSTANT", constant_values=False) # Pad False to the left
            sorted_indices_to_remove_mask = mask_with_false_prepended[:, :-1] # Slice to original size, first element is now effectively False


            # 5. Convert the removal mask from sorted space back to original vocabulary space
            sorted_indices_to_keep_mask = tf.logical_not(sorted_indices_to_remove_mask)

            batch_size_tf = tf.shape(predictions)[0]
            vocab_size_tf = tf.shape(predictions)[1]

            batch_idx_flat = tf.repeat(tf.range(batch_size_tf), vocab_size_tf) 
            original_vocab_idx_flat = tf.reshape(sorted_indices, [-1]) 
            
            scatter_update_indices = tf.stack([batch_idx_flat, original_vocab_idx_flat], axis=1) 

            flat_keep_values_for_sorted_pos = tf.reshape(sorted_indices_to_keep_mask, [-1]) 

            original_space_keep_mask = tf.scatter_nd(
                indices=scatter_update_indices,
                updates=flat_keep_values_for_sorted_pos,
                shape=tf.shape(predictions, out_type=tf.int32) 
            )
            
            filtered_logits = tf.where(
                original_space_keep_mask, 
                predictions,              
                tf.fill(tf.shape(predictions), -float('inf')) 
            )
            
        else: # No top_p filtering
            filtered_logits = predictions
        
        # Ensure filtered_logits are float32 for categorical, and add diagnostics
        filtered_logits = tf.cast(filtered_logits, tf.float32)
        
        tf.print("Shape of filtered_logits:", tf.shape(filtered_logits), output_stream=sys.stderr)
        # tf.print("Filtered logits for sampling (first batch item, first 10 vocab):", filtered_logits[0, :10], output_stream=sys.stderr) # Might be too verbose
        finite_logits_exist_per_row = tf.reduce_any(tf.math.is_finite(filtered_logits), axis=-1)
        tf.print("Finite logits exist per row:", finite_logits_exist_per_row, output_stream=sys.stderr)
        if not tf.reduce_all(finite_logits_exist_per_row):
            tf.print("WARNING: Some rows in filtered_logits have NO finite values!", output_stream=sys.stderr)
            # For rows where all logits are -inf, tf.random.categorical might behave unpredictably or error.
            # As a fallback, can make all logits 0 for these rows (uniform sampling).
            # This situation should ideally be prevented by the "keep at least one token" logic.
            # Example fallback:
            # uniform_fallback_logits = tf.zeros_like(filtered_logits)
            # filtered_logits = tf.where(finite_logits_exist_per_row[:, tf.newaxis], filtered_logits, uniform_fallback_logits)
            # tf.print("Applied uniform fallback for rows with all -inf logits.", output_stream=sys.stderr)


        predicted_ids_tensor = tf.random.categorical(filtered_logits, num_samples=1) # Shape (batch_size, 1), dtype=tf.int64
        predicted_ids_tensor_int32 = tf.cast(predicted_ids_tensor, dtype=tf.int32) # Cast to tf.int32
        
        # Append the new predicted token to input_eval for the next iteration
        input_eval = tf.concat([input_eval, predicted_ids_tensor_int32], axis=1)
        
        # Add the character to the generated text list (assuming batch_size=1 for generation)
        text_generated.append(index_to_char[predicted_ids_tensor_int32[0,0].numpy()])
    
    return (start_string + ''.join(text_generated))

# Metric functions (adapted from test.py)
def compute_spelling_accuracy(text, word_list):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    correct = sum(w in word_list for w in words)
    return correct / len(words)

def ngram_overlap(generated_text, reference_text, n=2):
    def get_ngrams(text, n_gram):
        # Ensure text is a string and handle potential non-string items if they could occur
        text_str = str(text)
        return set([text_str[i:i+n_gram] for i in range(len(text_str)-n_gram+1)])
    gen_ngrams = get_ngrams(generated_text, n)
    ref_ngrams = get_ngrams(reference_text, n)
    if not gen_ngrams:
        return 0.0
    return len(gen_ngrams & ref_ngrams) / len(gen_ngrams)

def ngram_diversity(text, n=2):
    # Ensure text is a string
    text_str = str(text)
    ngrams = [text_str[i:i+n] for i in range(len(text_str)-n+1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

if __name__ == "__main__":
    # train_transformer now returns test_text_str
    transformer, char_to_index, index_to_char, test_set_reference_text = train_transformer()
    
    print("\n--- Generating Sample Text for Metrics (using Nucleus Sampling) ---")
    start_string_for_metrics = "ROMEO." 
    # Generate text using nucleus sampling by default (top_p=0.9 is default in function)
    generated_text_sample = generate_text(transformer, start_string_for_metrics, char_to_index, index_to_char, generation_length=300)
    
    print(f"\nGenerated Sample:\n{generated_text_sample}")

    print("\n--- Calculating Metrics (on Test Set) ---")
    # Use the test_set_reference_text for metrics
    word_list_for_metrics = set(re.findall(r'\b\w+\b', test_set_reference_text.lower()))

    spelling_acc = compute_spelling_accuracy(generated_text_sample, word_list_for_metrics)
    # N-gram overlap compares generated text with the test set reference text
    bigram_ov = ngram_overlap(generated_text_sample, test_set_reference_text, n=2)
    trigram_div = ngram_diversity(generated_text_sample, n=3)
    bigram_div = ngram_diversity(generated_text_sample, n=2)

    print(f"\nMetrics for Transformer Model (evaluated against Test Set):")
    print(f"Spelling Accuracy (words from test set): {spelling_acc:.2f}")
    print(f"Bigram Overlap with Test Set: {bigram_ov:.2f}")
    print(f"Trigram Diversity of Generated Sample: {trigram_div:.2f}")
    print(f"Bigram Diversity of Generated Sample: {bigram_div:.2f}") 