import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-u', help='GPUs # to train on', type=str, default='7')
args = parser.parse_args()

print(args.gpu)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


from jiwer import wer
import nltk
import pickle
import textdistance
import pandas as pd

with open('deepspeech2_output_text_nonoise100%+timestretch100%', 'rb') as outputfile:
    output_text=pickle.load(outputfile)
with open('deepspeech2_beam_prob_nonoise100%+timestretch100%', 'rb') as outputfile:
    beam_prob=pickle.load(outputfile)
with open('deepspeech2_targets_nonoise+timestretch', 'rb') as outputfile:
    targets=pickle.load(outputfile)



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


best_word_error_rate = 100

for alpha in range(0,200,2):
    alpha = 0.01*alpha

    for beta in range(0,200,4):
        beta = 0.1*beta

        best_predictions=get_best_predictions(output_text)
        best_predictions_corrected = []
        for line in best_predictions:
            corrected_line = get_corrected_sentence(line)
            best_predictions_corrected.append(corrected_line)

        word_error_rate = wer(targets, best_predictions_corrected)
        if word_error_rate < best_word_error_rate:
            best_word_error_rate = word_error_rate
            best_alpha = alpha
            best_beta = beta

        print('alpha=', alpha, 'beta=', beta, 'wer=', word_error_rate)
        print('best alpha=', best_alpha, 'best beta=', best_beta, 'best_word_error_rate=', best_word_error_rate)
        print('-'*100)

