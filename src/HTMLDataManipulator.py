from bs4 import BeautifulSoup
import sys
import os

os.chdir('../data/SentimentStockCorrelation/figma_htmlgenerator')


spans = ['e2_24', 'e2_25', 'e2_26', 'e2_26', 'e2_27']
data = {'e2_24': 'Tesla', 'e2_25': 'Something something', 'e2_26': 'Something something 2', 'e5_26': 'e5_27', 'Spam': '83'}

with open('index.html', 'r') as file:
    htmlFile = file.read()
    soup = BeautifulSoup(htmlFile)

for elm in soup.findAll("span"):
    if str(elm['class']) in data.keys():
        elm.replace(data[elm['class']])

for elm in soup.findAll("div"):
    if str(elm['class']) in data.keys():
        elm.replace(data[elm['class']])

with open("output1.html", "w") as file:
    file.write(str(soup))