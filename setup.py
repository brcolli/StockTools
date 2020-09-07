from setuptools import setup

setup(
    name='StockTools',
    version='0.1',
    packages=[],
    url='',
    license='',
    author='benrc',
    author_email='',
    description='',
    install_requires=['yahoo-earnings-calendar', 'yfinance', 'tweepy', 'pandas', 'datetime', 'PyQt5'],
    scripts=['UpcomingEarningsScanner/UpcomingEarningsScanner.py', 'NewsSentimentAnalysis/TwitterSentimentAnalysis.py']
)
