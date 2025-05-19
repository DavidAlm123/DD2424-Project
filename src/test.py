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
import re
import numpy as np
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR =  BASE_DIR / "Project_RNN" / "shaketext.txt"
fname = DATASET_DIR

with open(fname, "r") as fid:
    data = fid.read()

unique_chars = list(set(data))
K = len(unique_chars)
unique_chars_sorted = sorted(unique_chars)

char_to_index = {char: index for index, char in enumerate(unique_chars_sorted)}
index_to_char = {index: char for index, char in enumerate(unique_chars_sorted)}

print("Total characters:", len(data))
print("Unique characters (K):", K)
print("Sample char to index mapping:", list(char_to_index.items())[:10])

text_as_int = [char_to_index[c] for c in data]

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

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)


for x_batch, y_batch in train_dataset.take(10):
    print("Input batch shape:", x_batch.shape)
    print("Target batch shape:", y_batch.shape)
    first_input = ''.join(index_to_char[idx] for idx in x_batch[0].numpy())
    first_target = ''.join(index_to_char[idx] for idx in y_batch[0].numpy())
    print("Decoded input:", first_input)
    print("Decoded target:", first_target)
    break

rnn_units = 100
embedding_dim = rnn_units//2

model = tf.keras.Sequential([
    layers.Embedding(input_dim=K, output_dim=embedding_dim),
    layers.SimpleRNN(rnn_units, return_sequences=True),
    layers.Dense(K)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn)

def sample(model, start_string, generation_length=500, temperature=1.0):
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    generated = []

    for _ in range(generation_length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        generated.append(index_to_char[predicted_id])

    return start_string + ''.join(generated)

print('Generated text pre-training:')
print(sample(model, start_string=data[:6], generation_length=300))
print()

EPOCHS = 10
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

model.save_weights('BASE_DIR/"Project_RNN".h5')

print()
print('Generated text post-training:')
print(sample(model, start_string=data[:6], generation_length=300))

m = rnn_units 
rng = np.random.default_rng()
BitGen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = BitGen(seed).state

def initialize_lstm(L, m):
    layers = []
    for l in range(L):
        in_dim = K if l == 0 else m
        layer = {
          'Wf': (1/np.sqrt(2*m))*rng.standard_normal((m, m)),
          'Uf': (1/np.sqrt(2*in_dim))*rng.standard_normal((m, in_dim)),
          'bf': np.zeros((m,1)),
          'Wi': (1/np.sqrt(2*m))*rng.standard_normal((m, m)),
          'Ui': (1/np.sqrt(2*in_dim))*rng.standard_normal((m, in_dim)),
          'bi': np.zeros((m,1)),
          'Wc': (1/np.sqrt(2*m))*rng.standard_normal((m, m)),
          'Uc': (1/np.sqrt(2*in_dim))*rng.standard_normal((m, in_dim)),
          'bc': np.zeros((m,1)),
          'Wo': (1/np.sqrt(2*m))*rng.standard_normal((m, m)),
          'Uo': (1/np.sqrt(2*in_dim))*rng.standard_normal((m, in_dim)),
          'bo': np.zeros((m,1))
        }
        layers.append(layer)
    # output layer
    V = (1/np.sqrt(m))*rng.standard_normal((K, m))
    c = np.zeros((K,1))
    return {'layers': layers, 'V': V, 'c': c}

def fp_lstm(RNN, X, Y, h0s, c0s):
    layers = RNN['layers']
    V, c_out = RNN['V'], RNN['c']
    L = len(layers)

    # prepare storages
    f = [ [None]*seq_length for _ in range(L) ]
    i = [ [None]*seq_length for _ in range(L) ]
    c_tilde = [ [None]*seq_length for _ in range(L) ]
    o = [ [None]*seq_length for _ in range(L) ]
    c_states = [ [None]*(seq_length+1) for _ in range(L) ]
    h_states = [ [None]*(seq_length+1) for _ in range(L) ]
    y_logits, p = [], []

    # init
    for l in range(L):
        h_states[l][0] = h0s[l]
        c_states[l][0] = c0s[l]

    loss = 0
    # time-step loop
    for t in range(seq_length):
        # input to layer 0
        x_in = X[:, t:t+1]    # shape (K,1)

        # forward through L layers
        for l in range(L):
            Wf, Uf, bf = layers[l]['Wf'], layers[l]['Uf'], layers[l]['bf']
            Wi, Ui, bi = layers[l]['Wi'], layers[l]['Ui'], layers[l]['bi']
            Wc, Uc, bc = layers[l]['Wc'], layers[l]['Uc'], layers[l]['bc']
            Wo, Uo, bo = layers[l]['Wo'], layers[l]['Uo'], layers[l]['bo']

            h_prev = h_states[l][t]
            c_prev = c_states[l][t]

            # gates
            fl = 1/(1+np.exp(-(Wf@h_prev + Uf@x_in + bf)))
            il = 1/(1+np.exp(-(Wi@h_prev + Ui@x_in + bi)))
            cbarl = np.tanh(   Wc@h_prev + Uc@x_in + bc)
            cl = fl*c_prev + il*cbarl
            ol = 1/(1+np.exp(-(Wo@h_prev + Uo@x_in + bo)))
            hl = ol * np.tanh(cl)

            # stash
            f[l][t] = fl;    i[l][t] = il
            c_tilde[l][t] = cbarl; o[l][t] = ol
            c_states[l][t+1] = cl; h_states[l][t+1] = hl

            # next layer’s input
            x_in = hl

        # output & loss
        logit = V @ h_states[L-1][t+1] + c_out
        exp_l = np.exp(logit - np.max(logit))
        p_t = exp_l/np.sum(exp_l)
        loss += -np.log(Y[:,t:t+1].T @ p_t)

        y_logits.append(logit); p.append(p_t)

    cache = {
      'f': f, 'i': i, 'c_tilde': c_tilde, 'o': o,
      'c_states': c_states, 'h_states': h_states,
      'y_logits': y_logits, 'p': p
    }

    return cache, loss[0,0]/seq_length

def bp_lstm(RNN, X, Y, cache):
    layers = RNN['layers']
    V, c_out = RNN['V'], RNN['c']
    L = len(layers)

    # unpack cache
    f = cache['f']; i = cache['i']; cbar = cache['c_tilde']; o = cache['o']
    c_states = cache['c_states']; h_states = cache['h_states']
    p = cache['p']

    # prepare gradients
    grads = { 'layers': [], 'V': np.zeros_like(V), 'c': np.zeros_like(c_out) }
    for l in range(L):
        in_dim = K if l == 0 else m
        grads['layers'].append({
          'Wf':np.zeros((m,m)), 'Uf':np.zeros((m,in_dim)), 'bf':np.zeros((m,1)),
          'Wi':np.zeros((m,m)), 'Ui':np.zeros((m,in_dim)), 'bi':np.zeros((m,1)),
          'Wc':np.zeros((m,m)), 'Uc':np.zeros((m,in_dim)), 'bc':np.zeros((m,1)),
          'Wo':np.zeros((m,m)), 'Uo':np.zeros((m,in_dim)), 'bo':np.zeros((m,1))
        })

    # time-step and BPTT buffers per layer
    dh_time = [np.zeros((m,1)) for _ in range(L)]
    dc_time = [np.zeros((m,1)) for _ in range(L)]

    # reverse time loop
    for t in reversed(range(seq_length)):
        # — output layer —
        dy = p[t] - Y[:,t:t+1]
        grads['V'] += dy @ h_states[L-1][t+1].T
        grads['c'] += dy

        # seed up-stream grad into top layer
        dh_time[L-1] += V.T @ dy

        # now backprop through layers l=L-1…0
        dh_down = None
        for l in reversed(range(L)):
            Wf, Uf, bf = layers[l]['Wf'], layers[l]['Uf'], layers[l]['bf']
            Wi, Ui, bi = layers[l]['Wi'], layers[l]['Ui'], layers[l]['bi']
            Wc, Uc, bc = layers[l]['Wc'], layers[l]['Uc'], layers[l]['bc']
            Wo, Uo, bo = layers[l]['Wo'], layers[l]['Uo'], layers[l]['bo']

            # gather forward caches
            fl = f[l][t]; il = i[l][t]; cbarl = cbar[l][t]; ol = o[l][t]
            c_prev = c_states[l][t]; c_curr = c_states[l][t+1]
            h_prev = h_states[l][t]; h_curr = h_states[l][t+1]

            # total dh into this layer = dh from next time-step + dh from above-layer
            dh = dh_time[l]
            # dc from next time-step
            dc = dc_time[l]

            # --- gate gradients ---
            dao = dh * np.tanh(c_curr)
            dao_raw = dao * ol*(1-ol)

            dc_tot = dh*ol*(1-np.tanh(c_curr)**2) + dc

            daf = dc_tot * c_prev
            daf_raw = daf * fl*(1-fl)

            dai = dc_tot * cbarl
            dai_raw = dai * il*(1-il)

            dac = dc_tot * il
            dac_raw = dac * (1-cbarl**2)

            # accumulate grads
            # input to this layer at time t:
            x_in = X[:,t:t+1] if l==0 else h_states[l-1][t+1]

            gl = grads['layers'][l]
            gl['Wf'] += daf_raw @ h_prev.T
            gl['Uf'] += daf_raw @ x_in.T
            gl['bf'] += daf_raw

            gl['Wi'] += dai_raw @ h_prev.T
            gl['Ui'] += dai_raw @ x_in.T
            gl['bi'] += dai_raw

            gl['Wc'] += dac_raw @ h_prev.T
            gl['Uc'] += dac_raw @ x_in.T
            gl['bc'] += dac_raw

            gl['Wo'] += dao_raw @ h_prev.T
            gl['Uo'] += dao_raw @ x_in.T
            gl['bo'] += dao_raw

            # --- propagate to previous time-step for same layer ---
            dh_time[l] = (Wf.T@daf_raw +
                          Wi.T@dai_raw +
                          Wc.T@dac_raw +
                          Wo.T@dao_raw)
            dc_time[l] = dc_tot * fl

            # --- propagate down to layer l−1 at same time t ---
            if l>0:
                dh_time[l-1] += (
                  Uf.T@daf_raw +
                  Ui.T@dai_raw +
                  Uc.T@dac_raw +
                  Uo.T@dao_raw
                )

        # end per-layer loop
    # end time-loop

    # average over timesteps
    for l in range(L):
        for k in grads['layers'][l]:
            grads['layers'][l][k] /= seq_length
    grads['V'] /= seq_length
    grads['c'] /= seq_length

    return grads

def train_lstm_adam_with_tracking(train_dataset, val_dataset, init_RNN, params):

    eta = params['eta']
    num_epochs = params['num_epochs']
    beta1 = params['beta1']
    beta2 = params['beta2']
    eps = params['eps']
    l2 = params.get('l2', 0.0)
    dropout_rate = params.get('dropout', 0.0)
    patience = params.get('early_stopping_patience', 5)
    verbose = params.get('verbose', True)

    RNN = copy.deepcopy(init_RNN)

    L = len(RNN['layers'])
    m1 = {'layers': [], 'V': np.zeros_like(RNN['V']), 'c': np.zeros_like(RNN['c'])}
    v1 = {'layers': [], 'V': np.zeros_like(RNN['V']), 'c': np.zeros_like(RNN['c'])}
    for l in range(L):
        zero_buf = {k: np.zeros_like(RNN['layers'][l][k]) for k in RNN['layers'][l]}
        m1['layers'].append({**zero_buf})
        v1['layers'].append({**zero_buf})

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    def compute_accuracy(preds, Y):
        preds_idx = np.argmax(preds, axis=0)
        true_idx = np.argmax(Y, axis=0)
        return np.mean(preds_idx == true_idx)

    def dropout_mask(shape, rate):
        return (np.random.rand(*shape) > rate).astype(np.float32)

    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_seqs = 0

        for x_seq, y_seq in tqdm(train_dataset.unbatch(), desc=f"Epoch {epoch}", unit="seq"):
            x_ids = x_seq.numpy()
            y_ids = y_seq.numpy()

            X = np.zeros((K, seq_length))
            Y = np.zeros((K, seq_length))
            for t in range(seq_length):
                X[x_ids[t], t] = 1
                Y[y_ids[t], t] = 1

            h0s = [np.zeros((m,1)) for _ in range(L)]
            c0s = [np.zeros((m,1)) for _ in range(L)]

            cache, loss = fp_lstm(RNN, X, Y, h0s, c0s)

            # L2 regularization
            l2_loss = 0
            for l in range(L):
                for k in ['Wf', 'Uf', 'Wi', 'Ui', 'Wc', 'Uc', 'Wo', 'Uo']:
                    l2_loss += np.sum(RNN['layers'][l][k]**2)
            l2_loss += np.sum(RNN['V']**2)
            total_loss = loss + l2 * l2_loss

            grads = bp_lstm(RNN, X, Y, cache)

            epoch_loss += total_loss
            epoch_acc += compute_accuracy(np.stack(cache['p'], axis=1), Y)
            n_seqs += 1

            t = epoch * n_seqs
            for l in range(L):
                for param, g in grads['layers'][l].items():
                    g += l2 * RNN['layers'][l][param]
                    m1['layers'][l][param] = beta1*m1['layers'][l][param] + (1-beta1)*g
                    v1['layers'][l][param] = beta2*v1['layers'][l][param] + (1-beta2)*(g*g)
                    m_hat = m1['layers'][l][param] / (1 - beta1**t)
                    v_hat = v1['layers'][l][param] / (1 - beta2**t)
                    RNN['layers'][l][param] -= eta * m_hat / (np.sqrt(v_hat) + eps)

                    # dropout
                    if dropout_rate > 0 and 'W' in param and RNN['layers'][l][param].ndim == 2:
                        mask = dropout_mask(RNN['layers'][l][param].shape, dropout_rate)
                        RNN['layers'][l][param] *= mask

            for key in ('V','c'):
                g = grads[key] + (l2 * RNN[key] if key == 'V' else 0)
                m1[key] = beta1*m1[key] + (1-beta1)*g
                v1[key] = beta2*v1[key] + (1-beta2)*(g*g)
                m_hat = m1[key] / (1 - beta1**t)
                v_hat = v1[key] / (1 - beta2**t)
                RNN[key] -= eta * m_hat / (np.sqrt(v_hat) + eps)

        avg_loss = epoch_loss / n_seqs
        avg_acc = epoch_acc / n_seqs

        # Validation loss
        val_loss, val_acc, val_seqs = 0.0, 0.0, 0
        for x_seq, y_seq in val_dataset.unbatch():
            x_ids = x_seq.numpy()
            y_ids = y_seq.numpy()

            X = np.zeros((K, seq_length))
            Y = np.zeros((K, seq_length))
            for t in range(seq_length):
                X[x_ids[t], t] = 1
                Y[y_ids[t], t] = 1

            h0s = [np.zeros((m,1)) for _ in range(L)]
            c0s = [np.zeros((m,1)) for _ in range(L)]

            cache, loss = fp_lstm(RNN, X, Y, h0s, c0s)
            val_loss += loss
            val_acc += compute_accuracy(np.stack(cache['p'], axis=1), Y)
            val_seqs += 1

        avg_val_loss = val_loss / val_seqs
        avg_val_acc = val_acc / val_seqs

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_acc)
        history['val_acc'].append(avg_val_acc)

        if verbose:
            print(f"Epoch {epoch}/{num_epochs} — Train loss: {avg_loss:.4f}, acc: {avg_acc:.4f} | Val loss: {avg_val_loss:.4f}, acc: {avg_val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return RNN, history

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sample_text_lstm(RNN, start_string, generation_length, temperature=1.0):
    layers = RNN['layers']
    V, c_out = RNN['V'], RNN['c']
    L = len(layers)

    # 1) initialize hidden+cell for each layer to zeros
    h_states = [np.zeros((m,1)) for _ in range(L)]
    c_states = [np.zeros((m,1)) for _ in range(L)]

    # 2) prime with start_string (no sampling yet)
    last_char_id = None
    for ch in start_string:
        x = np.zeros((K,1))
        last_char_id = char_to_index[ch]
        x[last_char_id,0] = 1

        # run through each LSTM layer
        for l in range(L):
            Wf, Uf, bf = layers[l]['Wf'], layers[l]['Uf'], layers[l]['bf']
            Wi, Ui, bi = layers[l]['Wi'], layers[l]['Ui'], layers[l]['bi']
            Wc, Uc, bc = layers[l]['Wc'], layers[l]['Uc'], layers[l]['bc']
            Wo, Uo, bo = layers[l]['Wo'], layers[l]['Uo'], layers[l]['bo']

            h_prev = h_states[l]
            c_prev = c_states[l]

            f_gate   = sigmoid( Wf @ h_prev + Uf @ x + bf )
            i_gate   = sigmoid( Wi @ h_prev + Ui @ x + bi )
            c_tilde  =       np.tanh( Wc @ h_prev + Uc @ x + bc )
            c_curr   = f_gate * c_prev + i_gate * c_tilde
            o_gate   = sigmoid( Wo @ h_prev + Uo @ x + bo )
            h_curr   = o_gate * np.tanh(c_curr)

            # stash and pass to next layer
            h_states[l] = h_curr
            c_states[l] = c_curr
            x = h_curr

    # 3) now generate new chars
    generated = []
    for _ in range(generation_length):
        # one-hot last char as input
        x = np.zeros((K,1))
        x[last_char_id,0] = 1

        # forward through L layers
        for l in range(L):
            Wf, Uf, bf = layers[l]['Wf'], layers[l]['Uf'], layers[l]['bf']
            Wi, Ui, bi = layers[l]['Wi'], layers[l]['Ui'], layers[l]['bi']
            Wc, Uc, bc = layers[l]['Wc'], layers[l]['Uc'], layers[l]['bc']
            Wo, Uo, bo = layers[l]['Wo'], layers[l]['Uo'], layers[l]['bo']

            h_prev = h_states[l]
            c_prev = c_states[l]

            f_gate   = sigmoid( Wf @ h_prev + Uf @ x + bf )
            i_gate   = sigmoid( Wi @ h_prev + Ui @ x + bi )
            c_tilde  =       np.tanh( Wc @ h_prev + Uc @ x + bc )
            c_curr   = f_gate * c_prev + i_gate * c_tilde
            o_gate   = sigmoid( Wo @ h_prev + Uo @ x + bo )
            h_curr   = o_gate * np.tanh(c_curr)

            h_states[l] = h_curr
            c_states[l] = c_curr
            x = h_curr

        # output layer + sampling
        logits = (V @ h_curr + c_out).flatten() / temperature
        exp_logits = np.exp(logits - np.max(logits))
        p = exp_logits / exp_logits.sum()

        # draw a sample
        last_char_id = rng.choice(np.arange(K), p=p)
        generated.append(index_to_char[last_char_id])

    return start_string + ''.join(generated)

init_lstm1 = initialize_lstm(1, 100)

print('Generated text pre-training:')
print(sample_text_lstm(init_lstm1, data[:6], 300))
print()

params = {
    'eta': 0.001,
    'num_epochs': 1,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-8,
    'l2': 1e-4,
    'dropout': 0.2,
    'early_stopping_patience': 2,
    'verbose': True
}

lstm1, history1 = train_lstm_adam_with_tracking(train_dataset, val_dataset, init_lstm1, params)

print('Generated text post-training:')
print(sample_text_lstm(lstm1, 'ROMEO.', 300))

plt.plot(history1['train_loss'], label='Train Loss')
plt.plot(history1['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss over Epochs")
plt.show()

def augment_text(text, noise_level=0.03):
    chars = list(text)
    n = len(chars)
    new_chars = []

    for i in range(n):
        if np.random.rand() < noise_level:
            op = random.choice(['swap', 'drop', 'duplicate'])
            if op == 'swap' and i < n - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == 'drop':
                continue
            elif op == 'duplicate':
                new_chars.append(chars[i])

        new_chars.append(chars[i])

    return ''.join(new_chars)

augmented_data = augment_text(data)

unique_chars = list(set(data))
unique_chars_sorted = sorted(unique_chars)
char_to_index = {char: index for index, char in enumerate(unique_chars_sorted)}
index_to_char = {index: char for char, index in char_to_index.items()}

text_as_int_augmented = [char_to_index[c] for c in augmented_data]

total_len = len(text_as_int_augmented)
train_end = int(0.7 * total_len)
val_end = int(0.85 * total_len)

train_text_aug = text_as_int_augmented[:train_end]
val_text_aug = text_as_int_augmented[train_end:val_end]
test_text_aug = text_as_int_augmented[val_end:]


print("Training LSTM on augmented data...")

char_train_dataset_aug = tf.data.Dataset.from_tensor_slices(train_text_aug)
char_val_dataset_aug = tf.data.Dataset.from_tensor_slices(val_text_aug)
char_test_dataset_aug = tf.data.Dataset.from_tensor_slices(test_text_aug)

sequences_train_aug = char_train_dataset_aug.batch(seq_length + 1, drop_remainder=True)
sequences_val_aug = char_val_dataset_aug.batch(seq_length + 1, drop_remainder=True)
sequences_test_aug = char_test_dataset_aug.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

train_dataset_aug = sequences_train_aug.map(split_input_target).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
val_dataset_aug = sequences_val_aug.map(split_input_target).batch(BATCH_SIZE, drop_remainder=True)
test_dataset_aug = sequences_test_aug.map(split_input_target).batch(BATCH_SIZE, drop_remainder=True)

init_lstm2 = initialize_lstm(1, 100)
lstm2, history2 = train_lstm_adam_with_tracking(train_dataset_aug, val_dataset_aug, init_lstm2, params)

def nucleus_sample(probs, p=0.9):
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
    def get_ngrams(text, n):
        return set([text[i:i+n] for i in range(len(text)-n+1)])
    gen_ngrams = get_ngrams(generated_text, n)
    ref_ngrams = get_ngrams(reference_text, n)
    if not gen_ngrams:
        return 0.0
    return len(gen_ngrams & ref_ngrams) / len(gen_ngrams)

def ngram_diversity(text, n=2):
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

print("\nGenerated text from LSTM trained on original data (nucleus sampling):")
print(sample_text_lstm_nucleus(lstm1, 'ROMEO.', 300))

print("\nGenerated text from LSTM trained on augmented data (nucleus sampling):")
print(sample_text_lstm_nucleus(lstm2, 'ROMEO.', 300))

reference_text = data  
word_list = set(re.findall(r'\b\w+\b', reference_text.lower()))

print("Quantitative Evaluation on Test Set:")
for name, model, test_dataset in [('Original', lstm1, test_dataset), ('Augmented', lstm2, test_dataset_aug)]:
    total_acc = 0
    total_ngrams = Counter()
    total_count = 0
    for x_batch, y_batch in test_dataset:
        for x_seq, y_seq in zip(x_batch.numpy(), y_batch.numpy()):
            pred_seq = []
            L = len(model['layers'])
            h0s = [np.zeros((m, 1)) for _ in range(L)]
            c0s = [np.zeros((m, 1)) for _ in range(L)]
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
            preds_idx = preds_idx.flatten()  
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

with open("shaketext.txt", encoding="utf-8") as f:
    data = f.read().lower()
    data = re.sub(r"[^a-zA-Z0-9\s]", "", data)
    sentences = [s.split() for s in re.split(r'[.!?]', data) if s.strip()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
index_word = {i: w for w, i in word_index.items()}
vocab_size = len(word_index) + 1
seq_length = 30

embedding_dim = 100
embedding_index = {}
with open("glove.6B.100d.txt", encoding='utf8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vec

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    vec = embedding_index.get(word)
    if vec is not None:
        embedding_matrix[i] = vec

tokens = tokenizer.texts_to_sequences(sentences)
sequences = []
for line in tokens:
    if len(line) >= seq_length + 1:
        for i in range(seq_length, len(line)):
            sequences.append(line[i-seq_length:i+1])
sequences = np.array(sequences)
np.random.shuffle(sequences)

train_end = int(0.7 * len(sequences))
val_end = int(0.85 * len(sequences))
X_train_glove, y_train_glove = sequences[:train_end, :-1], sequences[:train_end, -1]
X_val_glove, y_val_glove = sequences[train_end:val_end, :-1], sequences[train_end:val_end, -1]
X_test_glove, y_test_glove = sequences[val_end:, :-1], sequences[val_end:, -1]

model_glove = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=seq_length, trainable=True),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])
model_glove.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history_glove = model_glove.fit(
    X_train_glove, y_train_glove,
    validation_data=(X_val_glove, y_val_glove),
    batch_size=64,
    epochs=10
)

with open("shaketext_clean.txt", "w", encoding="utf-8") as f:
    f.write(data)

bpe_tokenizer = ByteLevelBPETokenizer()
bpe_tokenizer.train(files="shaketext_clean.txt", vocab_size=8000, min_frequency=2)
ids = bpe_tokenizer.encode(data).ids
sequences_bpe = [ids[i-seq_length:i+1] for i in range(seq_length, len(ids))]
sequences_bpe = np.array(sequences_bpe)
np.random.shuffle(sequences_bpe)

train_end = int(0.7 * len(sequences_bpe))
val_end = int(0.85 * len(sequences_bpe))
X_train_bpe, y_train_bpe = sequences_bpe[:train_end, :-1], sequences_bpe[:train_end, -1]
X_val_bpe, y_val_bpe = sequences_bpe[train_end:val_end, :-1], sequences_bpe[train_end:val_end, -1]
X_test_bpe, y_test_bpe = sequences_bpe[val_end:, :-1], sequences_bpe[val_end:, -1]
vocab_size_bpe = bpe_tokenizer.get_vocab_size()

model_bpe = Sequential([
    Embedding(vocab_size_bpe, 100, input_length=seq_length),
    LSTM(128),
    Dense(vocab_size_bpe, activation='softmax')
])
model_bpe.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history_bpe = model_bpe.fit(
    X_train_bpe, y_train_bpe,
    validation_data=(X_val_bpe, y_val_bpe),
    batch_size=64,
    epochs=10
)

def plot_metrics(history, title_prefix):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'{title_prefix} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(history_glove, "GloVe")
plot_metrics(history_bpe, "BPE")

def nucleus_sample(preds, top_p=0.9):
    preds = np.asarray(preds).astype("float64")
    sorted_indices = np.argsort(preds)[::-1]
    sorted_preds = preds[sorted_indices]
    cumulative_probs = np.cumsum(sorted_preds)
    cutoff = np.searchsorted(cumulative_probs, top_p)
    selected_indices = sorted_indices[:cutoff + 1]
    selected_probs = sorted_preds[:cutoff + 1]
    selected_probs = selected_probs / np.sum(selected_probs)
    return np.random.choice(selected_indices, p=selected_probs)

def generate_glove_text(model, tokenizer, seed_text, length=50, top_p=0.9):
    for _ in range(length):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence[-seq_length:]], maxlen=seq_length)
        preds = model.predict(sequence, verbose=0)[0]
        pred_id = nucleus_sample(preds, top_p)
        next_word = index_word.get(pred_id, "")
        seed_text += ' ' + next_word
    return seed_text

def generate_bpe_text(model, tokenizer, seed_text, length=50):
    ids = tokenizer.encode(seed_text).ids
    for _ in range(length):
        sequence = pad_sequences([ids[-seq_length:]], maxlen=seq_length)
        pred_id = np.argmax(model.predict(sequence, verbose=0))
        ids.append(pred_id)
    return tokenizer.decode(ids)

def compute_spelling_accuracy(text, word_list):
    words = re.findall(r'\b\w+\b', text.lower())
    correct = sum(w in word_list for w in words)
    return correct / len(words) if words else 0.0

def ngram_overlap(gen_text, ref_text, n=2):
    gen = set(gen_text[i:i+n] for i in range(len(gen_text) - n + 1))
    ref = set(ref_text[i:i+n] for i in range(len(ref_text) - n + 1))
    return len(gen & ref) / len(gen) if gen else 0.0

def ngram_diversity(text, n=2):
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

loss_glove, acc_glove = model_glove.evaluate(X_test_glove, y_test_glove)
loss_bpe, acc_bpe = model_bpe.evaluate(X_test_bpe, y_test_bpe)

ref_text = data
word_list = set(re.findall(r'\b\w+\b', ref_text.lower()))
gen_glove = generate_glove_text(model_glove, tokenizer, "romeo and juliet", 50)
gen_bpe = generate_bpe_text(model_bpe, bpe_tokenizer, "romeo and juliet", 50)

print(f"\nTest Accuracy — GloVe: {acc_glove:.4f}")
print(f"Test Accuracy — BPE: {acc_bpe:.4f}")

for name, text in [("GloVe", gen_glove), ("BPE", gen_bpe)]:
    spelling_acc = compute_spelling_accuracy(text, word_list)
    bigram_overlap = ngram_overlap(text, ref_text, n=2)
    trigram_div = ngram_diversity(text, n=3)
    print(f"\n{name} Model — Evaluation:")
    print(f"Spelling Accuracy: {spelling_acc:.2f}")
    print(f"Bigram Overlap: {bigram_overlap:.2f}")
    print(f"Trigram Diversity: {trigram_div:.2f}")
    print(f"Sample Output:\n{text[:500]}\n")



