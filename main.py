"""Main class"""
# Data transformation libraries
import pandas as pd
import matplotlib.pyplot as plt
import pdb
# Service modules
from Database.Database import Database
from Service.EntregasService import EntregasService

# Preprocess modules
from Process.PrincipalComponentAnalysis import PcaAnalysis

# Prediction modules
from Process.Prediction import Prediction

# Default arguments
from Aux.ArgumentsHelper import get_arguments

# Global variables
from config import PREPROCESS, PRELOAD, TRAIN, EVALUATE, PREDICT, VERBOSE, MODE, FIELD, GROUP, SEASONALITY

def predict_data(data_corpus):
    """Predict method"""
    if VERB >= 1:
        print('Starting predictor...')

    pred = Prediction(data_corpus, ARGS.mode or MODE, ARGS.data_field or FIELD, VERB)

    if ARGS.preprocess or PREPROCESS:
        pred.preprocess(GROUP, SEASONALITY, True, True, True)

    pred.init_model()

    preds = []

    if SEASONALITY is False:    
        if ARGS.preload or PRELOAD:
            pred.load_pretrained()
        if ARGS.train or TRAIN:
            pred.train()
        if ARGS.evaluate or EVALUATE:
            pred.evaluate()
        if ARGS.predict or PREDICT:
            pred.predict()
    
    if SEASONALITY is True:
        for x in ['seasonal', 'trend', 'resid']:
            pred.preprocess_seasonality(x, True, True, True)
            if ARGS.preload or PRELOAD:
                pred.load_pretrained()
            if ARGS.train or TRAIN:
                pred.train()
            if ARGS.evaluate or EVALUATE:
                pred.evaluate()
            if ARGS.predict or PREDICT:
                preds.append(pred.predict())
        pred.show_pred_seasonality(preds[0]*preds[1]*preds[2])

def pca(data_corpus):
    """PCA method"""
    pca_imp = PcaAnalysis(VERB, data_corpus)
    if VERB >= 2:
        print(data_corpus.shape)
    pca_imp.get_pca_components(2)
    pca_imp.show_analysis()

def main():
    """main method"""
    if ARGS.source == 'sql':
        db_conn = Database(VERB)
        conn = db_conn.connect()
        entregas_service = EntregasService(ARGS.groupBy, VERB, conn)
        data_corpus, _ = entregas_service.get_corpus(ARGS.process)
        db_conn.close()
    elif ARGS.source == "csv":
        data_corpus = pd.read_csv(filepath_or_buffer=ARGS.input_file)
        if VERB >= 2:
            data_corpus.plot()
            plt.show()

    # Prediction process
    predict_data(data_corpus)

if __name__ == "__main__":
    ARGS = get_arguments()
    VERB = max([ARGS.verbose, VERBOSE])
    main()
