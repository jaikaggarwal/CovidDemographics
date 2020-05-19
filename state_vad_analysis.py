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
        df_vad = pd.read_csv(STORAGE_DIR + 'lexicon.txt', delimiter='\t', header=0)
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

        self.goodness_of_fit(valence_model, valence, vad_embeddings)
        self.goodness_of_fit(arousal_model, arousal, vad_embeddings)
        self.goodness_of_fit(dominance_model, dominance, vad_embeddings)

        return valence_model, arousal_model, dominance_model

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

    def get_post_score(self, encoded_sen):
        #         sen_split = sentence_tokenize(text)
        # encoded_sen = model.encode(text)
        valence_pred_results = self.valence_model.get_prediction(encoded_sen)
        valence_pred_means = valence_pred_results.predicted_mean
        arousal_pred_results = self.arousal_model.get_prediction(encoded_sen)
        arousal_pred_means = arousal_pred_results.predicted_mean
        dominance_pred_results = self.dominance_model.get_prediction(encoded_sen)
        dominance_pred_means = dominance_pred_results.predicted_mean
        return valence_pred_means, arousal_pred_means, dominance_pred_means

    def get_week_score(self, dirname, week_file):
        # Get data stored for subreddit
        if f'{week_file}.pkl' in os.listdir(STORAGE_DIR):
            with open(STORAGE_DIR + f'{week_file}.pkl', "rb") as file:
                week_embeddings = pickle.load(file)
                print("USE SAVED WEEK EMBEDDINGS")
        else:
            # arousal_model = self.fit_beta_reg(arousal, vad_embeddings)
            data = self.get_file_data(dirname, week_file)
            valid_text = []
            for element in data:
                valid_text.extend(sentence_tokenize(element))
            week_embeddings = model.encode(valid_text)
            with open(STORAGE_DIR + f'{week_file}.pkl', "wb") as file:
                pickle.dump(week_embeddings, file)

        v_scores, a_scores, d_scores = get_post_score(week_embeddings)

        print(len(valid_text) == len(v_scores) == len(a_scores) == len(d_scores))
        return np.mean(v_scores), np.mean(a_scores), np.mean(d_scores), np.std(v_scores), np.std(a_scores), np.std(d_scores)

    def run(self):
        for i in range(3, 4):
            print(f"Region {i}")
            dirname = f"{STORAGE_DIR}Region{i}/"
            sorted_files = sort_files(dirname, cutoff=11)
            valence_means = []
            valence_stds = []
            arousal_means = []
            arousal_stds = []
            dominance_means = []
            dominance_stds = []
            for file in sorted_files:
                print(file)
                vmean, amean, dmean, vstd, astd, dstd = get_week_score(dirname, file)
                valence_means.append(vmean)
                valence_stds.append(vstd)
                arousal_means.append(amean)
                arousal_stds.append(astd)
                dominance_means.append(dmean)
                dominance_stds.append(dstd)

            with open("valence_means.pkl", "wb") as file:
                pickle.dump(valence_means, file)
            with open("arousal_means.pkl", "wb") as file:
                pickle.dump(arousal_means, file)
            with open("dominance_means.pkl", "wb") as file:
                pickle.dump(dominance_means, file)
            with open("valence_stds.pkl", "wb") as file:
                pickle.dump(valence_stds, file)
            with open("arousal_means.pkl", "wb") as file:
                pickle.dump(arousal_stds, file)
            with open("dominance_means.pkl", "wb") as file:
                pickle.dump(dominance_stds, file)

if __name__ == "__main__":
    if model.device.type != "cuda":
        print("DEVICE NOT USING CUDA", model.device)
        sys.exit(1)
    else:
        print("DEVICE USING CUDA", model.device)

