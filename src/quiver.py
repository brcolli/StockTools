import quiverquant
import utilities


Utils = utilities.Utils

ticker = ''
politician = ''

quiver = quiverquant.quiver('bf4263dfcfd881ece6d8ab45852fe9c27b4b5508')

if ticker == '' and politician == '':
    data = quiver.congress_trading()
elif politician == '':
    data = quiver.congress_trading(ticker)
else:
    data = quiver.congress_trading(politician, politician=True)

print(data)
#Utils.write_dataframe_to_csv(data, '../data/')
