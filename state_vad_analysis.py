import os
import datetime
import sys
import time
import requests
from datetime import datetime, timedelta
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import pickle

EARLIEST_DATE = datetime(2019, 12, 19)
model = SentenceTransformer('bert-large-nli-mean-tokens')
STORAGE_DIR = '/ais/hal9000/jai/'
class VadAnalysis:

    def __init__(self):
        self.valence_model, self.arousal_model, self.dominance_model = self.init_reg_models()

    def init_reg_models(self):

        stored_files = os.listdir(STORAGE_DIR)
        df_vad = pd.read_csv('Vad Lexicon/lexicon.txt', delimiter='\t', header=0)
        df_vad = df_vad.dropna()
        df_vad.index = df_vad['Word']
        df_vad = df_vad[['Valence', 'Arousal', 'Dominance']]
        vad_words = list(df_vad.index)
        valence = np.array(df_vad['Valence'].tolist())
        arousal = np.array(df_vad['Arousal'].tolist())
        dominance = np.array(df_vad['Dominance'].tolist())

        ## Get vad embeddings from storage in case one of the models isn't pickled
        if 'vad_embeddings.pkl' in stored_files:
            with open(STORAGE_DIR + 'vad_embeddings.pkl', "rb") as file:
                vad_embeddings = pickle.load(file)
        else:
            vad_embeddings = np.array(model.encode(vad_words))
            with open(STORAGE_DIR + "vad_embeddings.pkl", "wb") as file:
                pickle.dump(vad_embeddings, file)

        if 'valence_model.pkl' in stored_files:
            with open(STORAGE_DIR + "valence_model.pkl", "rb") as file:
                valence_model = pickle.load(file)
        else:
            valence_model = self.fit_beta_reg(valence, vad_embeddings)
            with open(STORAGE_DIR + "valence_model.pkl", "wb") as file:
                pickle.dump(valence_model, file)

        if 'arousal_model.pkl' in stored_files:
            with open(STORAGE_DIR + "arousal_model.pkl", "rb") as file:
                arousal_model = pickle.load(file)
        else:
            arousal_model = self.fit_beta_reg(arousal, vad_embeddings)
            with open(STORAGE_DIR + "arousal_model.pkl", "wb") as file:
                pickle.dump(arousal_model, file)

        if 'dominance_model.pkl' in stored_files:
            with open(STORAGE_DIR + "dominance_model.pkl", "rb") as file:
                dominance_model = pickle.load(file)
        else:
            dominance_model = self.fit_beta_reg(dominance, vad_embeddings)
            with open(STORAGE_DIR + "dominance_model.pkl", "wb") as file:
                pickle.dump(dominance_model, file)


        # if 'arousal_model.pkl' in stored_files:
        #     arousal_model = pickle.load(STORAGE_DIR + "arousal_model.pkl")
        # else:
        #     arousal_model = self.fit_beta_reg(arousal, vad_embeddings)
        #
        # if 'dominance_model.pkl' in stored_files:
        #     dominance_model = pickle.load(STORAGE_DIR + "dominance_model.pkl")
        # else:
        #     dominance_model = self.fit_beta_reg(dominance, vad_embeddings)

        self.goodness_of_fit(valence_model, valence, vad_embeddings)
        self.goodness_of_fit(arousal_model, arousal, vad_embeddings)
        self.goodness_of_fit(dominance_model, dominance, vad_embeddings)

        return

    def fit_beta_reg(self, y, X):
        binom_glm = sm.GLM(y, X, family=sm.families.Binomial())
        binom_results = binom_glm.fit()
        return binom_results


    def goodness_of_fit(self, model, true, X):
        y_predicted = model.get_prediction(X)
        pred_vals = y_predicted.summary_frame()['mean']
        print(r2_score(true, pred_vals))


    def sentence_tokenize(self, text):
        return sent_tokenize(text)


    def none_or_empty(self, text):
        return text is None or len(text) == 0 or text == "[removed]" or text == '[deleted]'


    def get_dates(self, latest_date):
        # Get all week increments from earliest date until present
        dates = []
        epochs = []
        curr = EARLIEST_DATE
        while curr < latest_date:
            dates.append(curr.strftime("%m/%d/%Y"))
            epochs.append(int(curr.timestamp()))
            curr += timedelta(weeks=1)
        dates.append(latest_date.strftime("%m/%d/%Y"))
        epochs.append(int(latest_date.timestamp()))
        return dates, epochs


    def get_file_data(self, dirname, file_name):
        with open(f'{dirname}{file_name}', 'r', encoding="UTF-8") as file:
            data = file.readlines()
            return [line.strip() for line in data]


    def sort_files(self, dirname, cutoff=1):
        sort_map = {}
        max_int = cutoff
        for file in os.listdir(dirname):
            dot_index = file.index(".")
            week = int(file[4:dot_index])
            if week >= cutoff:
                if sort_map.get(week) is None:
                    sort_map[week] = [file]
                else:
                    sort_map[week].append(file)
            if week >= max_int:
                max_int = week
        sorted_files = []
        for i in range(cutoff, max_int + 1):
            sorted_files.extend(sort_map[i])
        return sorted_files


    def get_week_score(self, dirname, week_file):
        # Get data stored for subreddit

        data = self.get_file_data(dirname, week_file)
        all_means = np.array([])

        def get_post_score(text):
            #         sen_split = sentence_tokenize(text)
            encoded_sen = model.encode(text)
            valence_pred_results = valence_model.get_prediction(encoded_sen)
            valence_pred_means = valence_pred_results.predicted_mean
            arousal_pred_results = arousal_model.get_prediction(encoded_sen)
            arousal_pred_means = arousal_pred_results.predicted_mean
            dominance_pred_results = dominance_model.get_prediction(encoded_sen)
            dominance_pred_means = dominance_pred_results.predicted_mean
            return valence_pred_means, arousal_pred_means, dominance_pred_means

        valid_text = []
        for element in data:
            valid_text.extend(sentence_tokenize(element))
        v_scores, a_scores, d_scores = get_post_score(valid_text)
        print(len(valid_text) == len(v_scores) == len(a_scores) == len(d_scores))
        return np.mean(v_scores), np.mean(a_scores), np.mean(d_scores), np.std(v_scores), np.std(a_scores), np.std(d_scores)


if __name__ == "__main__":
    if model.device.type != "cuda":
        print("DEVICE NOT USING CUDA", model.device)
        sys.exit(1)
    else:
        print("DEVICE USING CUDA", model.device)

