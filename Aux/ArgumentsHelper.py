import argparse


def get_arguments():
    """

    command options and argument parsing

    """

    parser = argparse.ArgumentParser(description="ZARA PREDICTIONS",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--verbose','-v', default=0, type=int,
                        help='Nivel de salida de texto que se desea, por defecto %(default)d. 0 significa que'
                             'no hay salida de texto')

    data_miner_group = parser.add_argument_group('data miner')
    data_miner_group.add_argument('--source', '-f', type=str, default='csv', help='Procedencia datos [\'sql\' | \'csv\' ]')
    data_miner_group.add_argument('--input_file', '-i', type=str, default='example.csv', help='Localizacion del fichero, solo para  tipo de procedencia \'csv\'')
    data_miner_group.add_argument('--data_field', '-fi', type=str, help='Field name')

    sql_group = parser.add_argument_group('sql options')
    sql_group.add_argument('--process', '-p', type=str, help='Proceso SQL a ejecutar')
    sql_group.add_argument('--groupBy', '-g', type=str, help='Agrupación de la consulta a realizar')


    pred_group = parser.add_argument_group('predictor options')
    pred_group.add_argument('--mode', '-m', type=str, help='tipo de algoritmo, machine learning, o deep learning [\'machine\' | \'deep\']')
    pred_group.add_argument('--preprocess', '-prep', type=bool, help='Activate data preprocessing')
    pred_group.add_argument('--preload', '-prl', type=bool, help='Prelaod previous model')
    pred_group.add_argument('--train', '-tr', type=bool, help='Activate train model method')
    pred_group.add_argument('--predict', '-pred', type=bool, help='Activate prediction on model')
    args = parser.parse_args()

    return args
