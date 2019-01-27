from keras.models import load_model
from keras.preprocessing import sequence
import os
import pandas as pd
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='Path to file with reviews.')
parser.add_argument('--reviews_count', help='Path to file with reviews.')
args = parser.parse_args()

file_path = os.path.abspath(args.file)
match = re.match(r'(.*)\.([^.]+)$', file_path)
if match[1] and match[2]:
    out_file_path = f'{match[1]}.out.{match[2]}'
else:
    out_file_path = f'{file_path}.out'
review_count = args.reviews_count

print('Opening file...')
data = pd.read_csv(file_path)
data = data[len(data) - 5000:len(data) - 1]

print('Preparing data...')
# Filter bad values
data = data[data['review'].apply(lambda x: isinstance(x, str))]
data = data[data['stars'].apply(lambda x: not np.isnan(x))]
if review_count:
    data = data[0:int(args.reviews_count)]
data['review'] = data['review'].apply(lambda x: ''.join(re.findall('\w+', x)))
data['created_at'] = data['created_at'].apply(pd.to_datetime)
data['year'] = data['created_at'].apply(lambda x: x.year)
data['normalized_stars'] = data['stars'].apply(lambda x: (x - 1) / 4)

symbols = []
grouped_symbols = {}

for review in data['review']:
    symbols += list(review)

for symbol in symbols:
    if symbol in grouped_symbols:
        grouped_symbols[symbol] += 1
    else:
        grouped_symbols[symbol] = 1

sorted_words = sorted([(k, v) for k, v in grouped_symbols.items()], key=lambda x: x[1], reverse=True)
word2index = {}
index2word = {}

for idx, val in enumerate(sorted_words):
    index2word[idx] = val[0]
    word2index[val[0]] = idx

data['review_indexes'] = data['review'].apply(lambda x: list(map(lambda word: word2index[word], list(x))))

print('Loading model...')
model = load_model('models/jd_reviews_model.h5')

print('Prediction...')
review_indexes_padded = sequence.pad_sequences(data['review_indexes'], maxlen=500)
predicted_values = pd.Series()

for i in range(0, len(review_indexes_padded), 1000):
    print(i)
    predicted_values = predicted_values.append(pd.Series(model.predict(review_indexes_padded[i:i + 1000]).flatten()), ignore_index=True)

print('Saving results...')
result = pd.DataFrame(data={'review_id': [], 'stars': [], 'predicted_stars': []})
result['review_id'] = data['review_id']
result['stars'] = data['stars'].apply(int)
result['predicted_stars'] = predicted_values.apply(lambda x: round(x * 4 + 1)).to_list()
result.to_csv(out_file_path, index=False, header=True)

print(f'Output file saved in {out_file_path}')
print('Done!')
