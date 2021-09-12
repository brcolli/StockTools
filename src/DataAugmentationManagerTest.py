import unittest
import DataAugmentationManager as dam
import pandas as pd
pd.set_option('display.max_colwidth', None)


class MyTestCase(unittest.TestCase):

    def test_wrapper_no_inc_alpha(self):
        eda = dam.TweetEDA()

        sentences = [
            "Orange yellow and red bunnies",
            "15 3450 and some others",
            "http://www.google.com and another url inserted as http://www.youtube.com",
            "http://www.website.com plus some descriptive words and phrases",
            "The blue brown fox went to the river",
            "The blue brown fox went to the ocean to swim",
            "Text and more text and more text and more text",
            "Multiple sentences. Sentence two. Sentence three. Sentence four.",
            ]

        text_data = [{'text': s} for s in sentences]

        data_input = pd.DataFrame({'Tweet id': [1, 2, 3, 4, 5, 6, 7, 8], 'json': text_data})
        output = eda.wrapper(data_input, total_aug_to_make=16)
        print(output[['Tweet id', 'text']])
        self.assertEqual(len(output), 16)

    def test_wrapper_inc_alpha(self):
        eda = dam.TweetEDA()

        start_alpha = 0.1
        increment_alpha = 0.1
        rounds = 6

        sentences = [
            "It's always a good idea to seek shelter from the evil gaze of the sun.",
            "Flesh-colored yoga pants were far worse than even he feared.",
            "It took him a while to realize that everything he decided not to change, he was actually choosing.",
            "When he had to picnic on the beach, he purposely put sand in other people’s food.",
            "I'm a great listener, really good with empathy vs sympathy and all that, but I hate people.",
            "You should never take advice from someone who thinks red paint dries quicker than blue paint.",
            "While on the first date he accidentally hit his head on the beam.",
            "She was the type of girl that always burnt sugar to show she cared."
        ]

        text_data = [{'text': s} for s in sentences]

        data_input = pd.DataFrame({'Tweet id': [1, 2, 3, 4, 5, 6, 7, 8], 'json': text_data})
        output = eda.wrapper(data_input, total_aug_to_make=len(data_input)*rounds, increment_alpha=increment_alpha,
                             general_alpha=start_alpha)
        print(output[['Tweet id', 'text']])
        self.assertEqual(len(output), len(data_input)*rounds)


if __name__ == '__main__':
    unittest.main()
