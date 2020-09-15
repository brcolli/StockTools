from tkinter import Tk
from tkinter.filedialog import askopenfilename


def replace_line_to_comma(filename):

    input_name = filename.split('.')[0]
    output = input_name + '.csv'

    f1 = open(filename, 'r')

    data = f1.read()
    data = data.replace(',', '/')
    data = data.replace('|', ',')

    f1.close()

    f2 = open(output, 'w')
    f2.write(data)
    f2.close()


def main():

    Tk().withdraw()
    f_name = askopenfilename()

    replace_line_to_comma(f_name)


if __name__ == '__main__':
    main()
