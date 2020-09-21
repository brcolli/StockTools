from tkinter import Tk
from tkinter.filedialog import askopenfilename
from os import path
import importlib
import requests


Utils = importlib.import_module('utilities').Utils


# Gets a file from a simple URL/FILE format
def get_file_from_url(url, file):

    link = url + '/' + file
    r = requests.get(link, allow_redirects=True)

    return r.text


# Reformatting short interest file to a proper csv
def replace_line_to_comma(text):
    text = text.replace(',', '/')
    return text.replace('|', ',')


# From a selected file, load and reformat, then save to csv
def load_short_interest_text_and_write_to_csv(filename):

    sel_file = filename.split('/')[-1]
    input_name = sel_file.split('.')[0]
    output = '../data/' + input_name + '.csv'

    f1 = open(filename, 'r')

    data = f1.read()
    data = replace_line_to_comma(data)

    f1.close()

    f2 = open(output, 'w')
    f2.write(data)
    f2.close()


# Uses util function to reformat latest trading day
def get_latest_trading_day():
    ltd = Utils.get_proper_date_format(Utils.get_last_trading_day())
    return ltd.replace('-', '')


# Gets file from regsho consolidated short interest using a YYYYMMDD format and write to csv
def get_regsho_daily_short_to_csv(ymd):

    url = 'http://regsho.finra.org'
    filename = 'CNMSshvol' + ymd
    output = '../data/' + filename + '.csv'

    # Check if date already saved
    if path.exists(output):
        return

    data = get_file_from_url(url, filename + '.txt')
    text = replace_line_to_comma(data)

    with open(output, 'w') as f:
        f.write(text)


# Call function to write latest trading day's short interest to a csv
def get_latest_short_interest_data():
    ltd = get_latest_trading_day()
    get_regsho_daily_short_to_csv(ltd)


def import_short_interest_text_from_selection():
    Tk().withdraw()
    f_name = askopenfilename()

    load_short_interest_text_and_write_to_csv(f_name)


def main():
    get_latest_short_interest_data()


if __name__ == '__main__':
    main()
