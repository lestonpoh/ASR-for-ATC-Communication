import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-u', help='GPUs # to train on', type=str, default='7')
args = parser.parse_args()

print(args.gpu)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.keras import layers
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer
import os
import nltk
import pickle
import textdistance
import re
import IPython.display

metadata_path = '/home/lpoh003/deepspeech2/metadata_5.csv'
wavs_path = '/home/lpoh003/deepspeech2/resampled_audio/'

metadata_df = pd.read_csv(metadata_path)
metadata_df.drop(columns=metadata_df.columns[0], axis=1, inplace=True)
metadata_df = metadata_df.sample(frac=1,random_state=0).reset_index(drop=True)

split1 = int(len(metadata_df) * 0.85) #85% training set
split2 = int(len(metadata_df) * 0.95) #10% validation set 5% test set
# df_train = metadata_df[:split1]
# df_val = metadata_df[split1:split2]
df_test = metadata_df[split2:]

# df_train = pd.read_csv('/home/lpoh003/deepspeech2/metadata_nonoise+timestretch+pitchscale+whitenoise_train.csv')
# df_train = df_train.sample(frac=1,random_state=0).reset_index(drop=True)
#
#
# print(f"Size of the training set: {len(df_train)}")
# print(f"Size of the validation set: {len(df_val)}")
print(f"Size of the test set: {len(df_test)}")


characters = [x for x in "abcdefghijklmnopqrstuvwxyz'. "]
char_to_num = keras.layers.StringLookup(vocabulary=characters,oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

frame_length = 256
frame_step = 160
fft_length = 384

def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label

batch_size = 32
# Define the training dataset
# train_dataset = tf.data.Dataset.from_tensor_slices(
#     (list(df_train["wav_filename"]), list(df_train["transcript"]))
# )
# train_dataset = (
#     train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
#     .padded_batch(batch_size)
#     .prefetch(buffer_size=tf.data.AUTOTUNE)
# )
#
# # Define the validation dataset
# validation_dataset = tf.data.Dataset.from_tensor_slices(
#     (list(df_val["wav_filename"]), list(df_val["transcript"]))
# )
# validation_dataset = (
#     validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
#     .padded_batch(batch_size)
#     .prefetch(buffer_size=tf.data.AUTOTUNE)
# )

# Define the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_test["wav_filename"]), list(df_test["transcript"]))
)
test_dataset = (
    test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

model = keras.models.load_model('/home/lpoh003/deepspeech2/deepspeech2_150epoch_nonoise100%+timestretch100%/', compile=False)
# decoding
def ctc_decode(y_pred, input_length, greedy=True, beam_width=200,
               top_paths=200, merge_repeated=True):
    epsilon=1e-7
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + epsilon)
    input_length = tf.compat.v1.to_int32(input_length)

    if greedy:
        (decoded, log_prob) = ctc.ctc_greedy_decoder(
            inputs=y_pred,
            sequence_length=input_length)
    else:
        (decoded, log_prob) = ctc.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length, beam_width=beam_width,
            top_paths=top_paths, merge_repeated=merge_repeated)

    decoded_dense = [tf.compat.v1.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=-1)
                     for st in decoded]
    return (decoded_dense, log_prob)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=False,beam_width=beam_width, top_paths=beam_width,merge_repeated=False)
    return results

with open('deepspeech_languagemodel_nltk', 'rb') as fin:
    lm = pickle.load(fin)

def everygram_score(word1, word2, word3):
    return 0.5 * lm.logscore(word3, (word1 + ' ' + word2).split()) + 0.3 * lm.logscore(word3, word2.split()) + 0.2 * lm.logscore(word3)


def calculate_score(prediction):
    prediction = prediction.split()
    if len(prediction) == 0:
        return everygram_score('<s>', '<s>', ' ')

    for i in range(len(prediction)):
        if i == 0:
            score = everygram_score('<s>', '<s>', prediction[i])
        elif i == 1:
            score += everygram_score('<s>', prediction[i - 1], prediction[i])
        else:
            score += everygram_score(prediction[i - 2], prediction[i - 1], prediction[i])

    if len(prediction) == 1:
        score += everygram_score('<s>', prediction[-1], '</s>')
        score += everygram_score(prediction[-1], '</s>', '</s>')

    else:
        score += everygram_score(prediction[-2], prediction[-1], '</s>')
        score += everygram_score(prediction[-1], '</s>', '</s>')

    return score


def get_best_predictions(output_text):
    best_predictions = []
    for line in range(len(output_text)):

        max_score = beam_prob[line][0] \
                    + alpha * calculate_score(output_text[line][0]) \
                    + beta * len(output_text[line][0].split())
        best_predict = output_text[line][0]

        for count in range(1, len(output_text[line])):

            confidence_score = beam_prob[line][count] \
                               + alpha * calculate_score(output_text[line][count]) \
                               + beta * len(output_text[line][count].split())
            if confidence_score > max_score:
                best_predict = output_text[line][count]
                max_score = confidence_score

        best_predictions.append(best_predict)
    return best_predictions


# spelling corrector
with open('autocorrect_probability', 'rb') as fin:
    probs = pickle.load(fin)


def autocorrect(input_word):
    if input_word in probs.keys():
        pass
    else:
        sim = [1 - (textdistance.levenshtein.normalized_distance(v, input_word) + textdistance.Jaccard(qval=2).distance(
            v, input_word)) / 2 for v in probs.keys()]
        #sim = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in probs.keys()]
        # sim = [1 - (textdistance.levenshtein.normalized_distance(v, input_word)) for v in probs.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = sim
        output = df.sort_values(['Similarity', 'Prob'], ascending=False)[0:3]['Word']
        return output


def get_corrected_sentence(prediction):
    prediction = prediction.split()
    for i in range(len(prediction)):

        if prediction[i] not in probs.keys():
            if i == len(prediction) - 1:

                if i == 0:
                    word_score = (everygram_score('<s>', '<s>', prediction[i]) + everygram_score('<s>', prediction[i],
                                                                                                 '</s>')) / 2
                elif i == 1:
                    word_score = (everygram_score('<s>', prediction[i - 1], prediction[i]) + everygram_score(
                        prediction[i - 1], prediction[i], '</s>')) / 2
                else:
                    word_score = (everygram_score(prediction[i - 2], prediction[i - 1],
                                                  prediction[i]) + everygram_score(prediction[i - 1], prediction[i],
                                                                                   '</s>')) / 2

                for corrected_word in autocorrect(prediction[i]):
                    if i == 0:
                        corrected_word_score = (everygram_score('<s>', '<s>', corrected_word) + everygram_score('<s>',
                                                                                                                corrected_word,
                                                                                                                '</s>')) / 2
                    elif i == 1:
                        corrected_word_score = (everygram_score('<s>', prediction[i - 1],
                                                                corrected_word) + everygram_score(prediction[i - 1],
                                                                                                  corrected_word,
                                                                                                  '</s>')) / 2
                    else:
                        corrected_word_score = (everygram_score(prediction[i - 2], prediction[i - 1],
                                                                corrected_word) + everygram_score(prediction[i - 1],
                                                                                                  corrected_word,
                                                                                                  '</s>')) / 2

                    if corrected_word_score > word_score:
                        word_score = corrected_word_score
                        prediction[i] = corrected_word

            else:

                if i == 0:
                    word_score = (everygram_score('<s>', '<s>', prediction[i]) + everygram_score('<s>', prediction[i],
                                                                                                 prediction[i + 1])) / 2
                elif i == 1:
                    word_score = (everygram_score('<s>', prediction[i - 1], prediction[i]) + everygram_score(
                        prediction[i - 1], prediction[i], prediction[i + 1])) / 2
                else:
                    word_score = (everygram_score(prediction[i - 2], prediction[i - 1],
                                                  prediction[i]) + everygram_score(prediction[i - 1], prediction[i],
                                                                                   prediction[i + 1])) / 2

                for corrected_word in autocorrect(prediction[i]):
                    if i == 0:
                        corrected_word_score = (everygram_score('<s>', '<s>', corrected_word) + everygram_score('<s>',
                                                                                                                corrected_word,
                                                                                                                prediction[
                                                                                                                    i + 1])) / 2
                    elif i == 1:
                        corrected_word_score = (everygram_score('<s>', prediction[i - 1],
                                                                corrected_word) + everygram_score(prediction[i - 1],
                                                                                                  corrected_word,
                                                                                                  prediction[
                                                                                                      i + 1])) / 2
                    else:
                        corrected_word_score = (everygram_score(prediction[i - 2], prediction[i - 1],
                                                                corrected_word) + everygram_score(prediction[i - 1],
                                                                                                  corrected_word,
                                                                                                  prediction[
                                                                                                      i + 1])) / 2

                    if corrected_word_score > word_score:
                        word_score = corrected_word_score
                        prediction[i] = corrected_word
    return ' '.join(prediction)

best_word_error_rate = 100

alpha=1.12
beta=8

beam_width = 225
predictions = []
targets = []
for batch in test_dataset:
    X, y = batch

    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)

output_text = []
beam_prob = []
for i in range(0, len(predictions), 2):

    for line in range(len(predictions[i][0])):
        count = 0
        output_text_temp = []
        beam_prob_temp = []
        for j in predictions[i]:
            result = tf.strings.reduce_join(num_to_char(j[line])).numpy().decode("utf-8")
            result = result.strip()
            result = re.sub(' +', ' ', result)
            prob = predictions[i + 1][line].numpy()[count]
            if result not in output_text_temp:
                output_text_temp.append(result)
                beam_prob_temp.append(prob)
            else:
                idx = output_text_temp.index(result)
                beam_prob_temp[idx] = np.log10(10 ** beam_prob_temp[idx] + 10 ** prob)

            count += 1
        output_text.append(output_text_temp)
        beam_prob.append(beam_prob_temp)

best_predictions=get_best_predictions(output_text)
best_predictions_corrected = []
for line in best_predictions:
    corrected_line = get_corrected_sentence(line)
    best_predictions_corrected.append(corrected_line)

with open('deepspeech2_best_predictions_150epoch_nonoise100%+timestretch100%_225beam', "wb") as file_pi:
    pickle.dump(best_predictions_corrected,file_pi)
with open('deepspeech2_targets_150epoch_nonoise100%+timestretch100%_225beam', "wb") as file_pi:
    pickle.dump(targets,file_pi)



# for x in range(50,500,25):
#     beam_width=x
#     predictions = []
#     targets = []
#     for batch in test_dataset:
#         X, y = batch
#
#         batch_predictions = model.predict(X)
#         batch_predictions = decode_batch_predictions(batch_predictions)
#         predictions.extend(batch_predictions)
#         for label in y:
#             label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
#             targets.append(label)
#
#     output_text = []
#     beam_prob = []
#     for i in range(0, len(predictions), 2):
#
#         for line in range(len(predictions[i][0])):
#             count = 0
#             output_text_temp = []
#             beam_prob_temp = []
#             for j in predictions[i]:
#                 result = tf.strings.reduce_join(num_to_char(j[line])).numpy().decode("utf-8")
#                 result = result.strip()
#                 result = re.sub(' +', ' ', result)
#                 prob = predictions[i + 1][line].numpy()[count]
#                 if result not in output_text_temp:
#                     output_text_temp.append(result)
#                     beam_prob_temp.append(prob)
#                 else:
#                     idx = output_text_temp.index(result)
#                     beam_prob_temp[idx] = np.log10(10 ** beam_prob_temp[idx] + 10 ** prob)
#
#                 count += 1
#             output_text.append(output_text_temp)
#             beam_prob.append(beam_prob_temp)
#
#     best_predictions=get_best_predictions(output_text)
#     best_predictions_corrected = []
#     for line in best_predictions:
#         corrected_line = get_corrected_sentence(line)
#         best_predictions_corrected.append(corrected_line)
#
#     word_error_rate = wer(targets, best_predictions_corrected)
#     if word_error_rate < best_word_error_rate:
#         best_word_error_rate = word_error_rate
#         best_beam = x
#
#     print('beam_width=', x, 'wer=', word_error_rate)
#     print('best beam=', best_beam, 'best_word_error_rate=', best_word_error_rate)
#     print('-'*100)






