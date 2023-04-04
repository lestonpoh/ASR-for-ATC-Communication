import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import ctc_ops as ctc
import textdistance
import re
import pickle
from tensorflow import keras
import pyaudio
import wave

#load asr model
model = keras.models.load_model('deepspeech2_final_model', compile=False)


characters = [x for x in "abcdefghijklmnopqrstuvwxyz'. "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def get_spectrogram(audio_path):
    frame_length = 256
    frame_step = 160
    fft_length = 384
    file = tf.io.read_file(audio_path)
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    spectrogram=tf.expand_dims(spectrogram, 0)
    return spectrogram

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100,
               top_paths=100, merge_repeated=True):
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

beam_width=100

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=False,beam_width=beam_width, top_paths=100,merge_repeated=False)
    return results

#to create a list of all results and their beam probs with no duplicates
def get_beams(prediction):
    output_text=[]
    beam_prob=[]
    for i in range(len(prediction[0])):
        result = tf.strings.reduce_join(num_to_char(prediction[0][i])).numpy().decode("utf-8")
        result = result.strip()
        result = re.sub(' +',' ',result)
        prob = prediction[1][0].numpy()[i]
        if result not in output_text:
            output_text.append(result)
            beam_prob.append(prob)
        else:
            idx=output_text.index(result)
            beam_prob[idx]=np.log10(10**beam_prob[idx]+10**prob)
    return output_text,beam_prob


# language model
# load language model
with open('deepspeech_languagemodel_nltk', 'rb') as fin:
    lm = pickle.load(fin)

alpha = 1.12
beta = 8

def everygram_score(word1, word2, word3):
    return 0.5 * lm.logscore(word3, (word1 + ' ' + word2).split()) + 0.3 * lm.logscore(word3,word2.split()) + 0.2 * lm.logscore(word3)


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


def get_best_predictions(output_text,beam_prob):
    best_predictions = []
    max_score = beam_prob[0] \
                + alpha * calculate_score(output_text[0]) \
                + beta * len(output_text[0].split())
    best_predict = output_text[0]

    for count in range(1, len(output_text)):

        confidence_score = beam_prob[count] \
                           + alpha * calculate_score(output_text[count]) \
                           + beta * len(output_text[count].split())
        if confidence_score > max_score:
            best_predict = output_text[count]
            max_score = confidence_score

    return best_predict


# spelling corrector
with open('autocorrect_probability', 'rb') as fin:
    probs = pickle.load(fin)


def autocorrect(input_word):
    if input_word in probs.keys():
        pass
    else:
        sim = [1 - (textdistance.levenshtein.normalized_distance(v, input_word) + textdistance.Jaccard(qval=2).distance(
            v, input_word)) / 2 for v in probs.keys()]
        #         sim = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in probs.keys()]
        #         sim = [1-(textdistance.levenshtein.normalized_distance(v,input_word)) for v in probs.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = sim
        output = df.sort_values(['Similarity', 'Prob'], ascending=False)[0:3]['Word']
        return (output)


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


def get_prediction(audio_path):
    spectrogram = get_spectrogram(audio_path)
    prediction = model.predict_on_batch(spectrogram)
    prediction = decode_batch_predictions(prediction)
    output_text, beam_prob = get_beams(prediction)
    best_prediction = get_best_predictions(output_text,beam_prob)
    corrected_prediction = get_corrected_sentence(best_prediction)
    return corrected_prediction

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, input=True, frames_per_buffer=1024)

    frames = []
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:    
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()

    sound_file = wave.open("recorded_audio.wav","wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(22050)
    sound_file.writeframes(b''.join(frames))
    sound_file.close()