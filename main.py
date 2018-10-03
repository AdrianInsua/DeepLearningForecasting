"""Main class"""
# Data transformation libraries
import pandas as pd
import matplotlib.pyplot as plt
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
from config import PREPROCESS, PRELOAD, TRAIN, EVALUATE, PREDICT, VERBOSE, MODE, FIELD, GROUP, GROUP_BY, SEASONALITY

import pdb

def predict_data(data_corpus):
    """Predict method"""
    if VERB >= 1:
        print('Starting predictor...')
    train_data = data_corpus
    pred = Prediction(ARGS.mode or MODE, ARGS.data_field or FIELD, GROUP_BY, VERB)



    preds = []

    if SEASONALITY is False:    
        if ARGS.preprocess or PREPROCESS:
            data, scaler, train_data, test_data, pred_data = pred.preprocess(data_corpus, GROUP, SEASONALITY, True, True, True)
            pred.init_model(train_data['x'].shape)
        if ARGS.preload or PRELOAD:
            pred.load_pretrained()
        if ARGS.train or TRAIN:
            pred.train(train_data, test_data)
        if ARGS.evaluate or EVALUATE:
            preds = pred.evaluate(test_data, scaler)
        if ARGS.predict or PREDICT:
            preds = pred.predict(pred_data, scaler)
    
    if SEASONALITY is True:
        if ARGS.preprocess or PREPROCESS:
            data, dec_data, scaler, train_data, test_data, pred_data = pred.preprocess_seasonality(data_corpus, None, True, False, False, False)
            print(pred_data)
        for x in ['resid', 'trend', 'seasonal']:
            if ARGS.preprocess or PREPROCESS:
                data, dec_data, scaler, train_data, test_data, pred_data = pred.preprocess_seasonality(data, x, False, True, True, True)
                pred.init_model(train_data['x'].shape)
            if ARGS.preload or PRELOAD:
                pred.load_pretrained()
            if ARGS.train or TRAIN:
                pred.train(train_data, test_data)
            if ARGS.evaluate or EVALUATE:
                pred.evaluate(test_data, scaler)
            if ARGS.predict or PREDICT:
                preds.append(pred.predict(pred_data, scaler))
        preds_df = pd.concat([pd.DataFrame(preds[0]), pd.DataFrame(preds[1]), pd.DataFrame(preds[2])], axis=1)
        preds_df['y'] = preds_df.sum(axis=1);
        real = data_corpus[-12:]['IMPORTE'].reset_index()
        plt.figure()
        plt.plot(preds_df['y'], label='pred')
        plt.plot(real, label='true')
        plt.legend()
        plt.show()

    pdb.set_trace()

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
