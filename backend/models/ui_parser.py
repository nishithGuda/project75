import numpy as np
import re
import json
from collections import Counter


def simple_tokenize(text):
    """Basic tokenizer that strips punctuation and lowercases"""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


class UIFeatureExtractor:
    def __init__(self, max_vocab=1000, max_elements=10):
        self.max_vocab = max_vocab
        self.max_elements = max_elements
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        self.element_types = [
            'button', 'text', 'image', 'input', 'checkbox', 'radio',
            'dropdown', 'switch', 'layout', 'webview', 'view', 'unknown'
        ]

    def build_vocabulary(self, training_data):
        word_counts = Counter()

        for sample in training_data:
            word_counts.update(simple_tokenize(sample['query']))
            for elem in sample.get('elements', []):
                word_counts.update(simple_tokenize(elem.get('text', '')))
                word_counts.update(simple_tokenize(elem.get('content_desc', '')))

        most_common = word_counts.most_common(self.max_vocab - 2)
        for word, _ in most_common:
            self.word_to_idx[word] = self.vocab_size
            self.idx_to_word[self.vocab_size] = word
            self.vocab_size += 1

    def extract_query_features(self, query, max_length=15):
        tokens = simple_tokenize(query)
        indices = [self.word_to_idx.get(t, self.word_to_idx['<UNK>']) for t in tokens[:max_length]]
        if len(indices) < max_length:
            indices += [self.word_to_idx['<PAD>']] * (max_length - len(indices))
        return np.array(indices)

    def extract_element_features(self, element):
        features = []

        # Bag of Words from text + content_desc
        tokens = simple_tokenize((element.get('text') or '') + ' ' + (element.get('content_desc') or ''))
        bow = np.zeros(self.vocab_size)
        for token in tokens:
            bow[self.word_to_idx.get(token, self.word_to_idx['<UNK>'])] += 1
        features.extend(bow)

        # One-hot type
        onehot = np.zeros(len(self.element_types))
        etype = (element.get('type') or 'unknown').lower()
        if etype in self.element_types:
            onehot[self.element_types.index(etype)] = 1
        else:
            onehot[self.element_types.index('unknown')] = 1
        features.extend(onehot)

        # Positional features
        bounds = element.get('bounds', [0, 0, 0, 0])
        x1, y1, x2, y2 = bounds
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2 / 1440
        center_y = (y1 + y2) / 2 / 2560
        features.extend([center_x, center_y, width / 1440, height / 2560, (width * height) / (1440 * 2560)])

        # Binary features
        features.extend([
            1 if element.get('clickable', False) else 0,
            1 if element.get('enabled', True) else 0,
            1 if element.get('visible', True) else 0,
            1 if element.get('from_semantic', False) else 0
        ])

        # Depth
        features.append(element.get('depth', 0) / 10)

        return np.array(features)

    def process_sample(self, sample):
        query_vec = self.extract_query_features(sample['query'])
        element_vecs = []

        elements = sample.get('elements', [])[:self.max_elements]
        for elem in elements:
            element_vecs.append(self.extract_element_features(elem))

        # Pad with dummy vectors if needed
        if not element_vecs:
            dummy = np.zeros(self.vocab_size + len(self.element_types) + 10)
            element_vecs = [dummy] * self.max_elements
        else:
            while len(element_vecs) < self.max_elements:
                element_vecs.append(np.zeros_like(element_vecs[0]))

        element_vecs = np.array(element_vecs)

        # Build target vector
        target = np.zeros(self.max_elements)
        t_idx = sample.get('target_idx', -1)
        if 0 <= t_idx < len(elements):
            target[t_idx] = 1

        return {
            'query_features': query_vec,
            'element_features': element_vecs,
            'target': target,
            'screen_id': sample.get('screen_id', ''),
            'original_elements': elements
        }

    def process_mode_b_samples(self, sample):
        processed = self.process_sample(sample)
        entries = []
        for i, feature_vec in enumerate(processed['element_features']):
            entries.append({
                'query_features': processed['query_features'],
                'element_feature': feature_vec,
                'label': int(processed['target'][i])
            })
        return entries

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': {str(k): v for k, v in self.idx_to_word.items()},
                'vocab_size': self.vocab_size,
                'max_elements': self.max_elements,
                'max_vocab': self.max_vocab
            }, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        self.vocab_size = data['vocab_size']
        self.max_elements = data.get('max_elements', 10)
        self.max_vocab = data.get('max_vocab', 1000)
