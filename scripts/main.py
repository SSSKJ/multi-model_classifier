from prepocessing import preprocessor
from Model import Modeling

if __name__ == '__main__':

    df = preprocessor('bank-additional-full', 'processed_data')
    Modeling(df)
