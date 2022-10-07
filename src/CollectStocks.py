import yfinance as yf
import pandas as pd

#sp = pd.read_csv('sp-100.csv')
#tickers = sp['Symbol'].tolist()
#names = sp['Name'].tolist()

tickers = ['BRK-B']
names = ['Berkshire Hathaway']

for n, t in zip(names, tickers):
    a = yf.Ticker(t)
    x = a.history(start='2022-08-01', end='2022-08-31', interval='1d')
    x.to_csv(f'{n} PriceData.csv', index=False)



# input:
# Label (0 Pos or 1 Neg)
# Confidence of that Label (0.5-1.0)

def normal_score(label, confidence):
    # Positive
    if label == 0:
        norm_score = 50 + (confidence * 50)
        # High Confidence --> NormScore close to 100
        # Low Confidence --> NormScore close to 50

    # Negative
    else:
        norm_score = 50 - (confidence * 50)
        # High Confidence --> NormScore close to 0
        # Low Confidence --> NormScore close to 50

    return norm_score


# Get a label and confidence score including Neutral
def get_label_from_norm_score(norm_score):
    negative_cutoff = 34
    neutral_cutoff = 66

    if norm_score <= negative_cutoff:
        label = 0

        # How close was the norm_score to 0 (the given label), scaled to be between 0.0-1.0
        confidence = (abs(norm_score - 0)) / (negative_cutoff - 0)


    elif norm_score <= 66:
        label = 1

        # How close was the norm_score to 0.5 (the given label), scaled to be between 0.0-1.0
        confidence = (abs(norm_score - 0.5)) / ((neutral_cutoff - negative_cutoff)/2)


    else:
        label = 2

        # How close was the norm_score to 1 (the given label), scaled to be between 0.0-1.0
        confidence = (abs(norm_score - 1)) / (1 - neutral_cutoff)

    return label, confidence