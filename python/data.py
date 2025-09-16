import requests
import pandas as pd
import json

function = 'TIME_SERIES_WEEKLY'
symbol = 'YPF'
token = 'XHC6C76TRCXZWOW2'
size = 'compact'
urlbase = 'https://www.alphavantage.co/query?'

params = {
    'function': function,
    'symbol': symbol,
    'apikey': token,
    'size': size
}

r = requests.get(urlbase, params=params)

data = r.json()['Weekly Time Series']
dataDF = pd.DataFrame.from_dict(data, orient='index')
dataDF.index.name = 'Date'

print(dataDF)