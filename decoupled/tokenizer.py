import re
import json
from collections import Counter

class Tokenizer:
    def __init__(self, level: str = 'character', max_vocab_size: int = 1000, bpe_iterations: int=100, custom_tokens: list=None):
        self.level = level
        self.vocab = {}
        self.decode_vocab = {}
        self.vocab_size = max_vocab_size
        self.bpe_iterations = bpe_iterations
        self.custom_tokens = custom_tokens if custom_tokens else []

    def tokenize(self, text):
        if self.level == 'character':
            return self.character(text)
        elif self.level == 'word':
            return self.word(text)
        elif self.level == 'sub_word':
            return self.sub_word(text)
        else:
            raise ValueError("Invalid tokenization level. Choose 'character', 'word', or 'sub_word'.")


    def character(self, text):
        chars = list(text)
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
                self.decode_vocab[len(self.decode_vocab)] = char
        return [self.vocab[char] for char in chars]

    def word(self, text):
        words = text.split()
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                self.decode_vocab[len(self.decode_vocab)] = word
        return [self.vocab[word] for word in words]

    def sub_word(self, text):
      
        tokens = re.findall(r'\w+|[^\w\s]', text) #get initial tokens

        if not self.vocab: # build the vocab only if empty
          self._build_bpe_vocab(tokens)
          
        encoded_tokens = []
        for token in tokens:
            encoded_tokens.extend(self._encode_token(token)) #encode the tokens, using the bpe algorithm
           
        return encoded_tokens

    def _build_bpe_vocab(self, tokens):
        """Builds the vocabulary using BPE"""
        
        word_counts = Counter(tokens)
        
        # Add custom tokens at the beginning of the vocab if they do not exists
        for token in self.custom_tokens:
           if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.decode_vocab[len(self.decode_vocab)] = token
        
        # add initial tokens
        for token in word_counts.keys():
          if token not in self.vocab:
              self.vocab[token] = len(self.vocab)
              self.decode_vocab[len(self.decode_vocab)] = token
        
        
        if len(self.vocab) >= self.vocab_size: # If the initial tokens are already greater than the vocabulary size, then skip BPE
            return
        
        for _ in range(self.bpe_iterations):
            pairs = self._get_pairs(word_counts)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            new_vocab_token = ''.join(best_pair)
            if new_vocab_token not in self.vocab:
                self.vocab[new_vocab_token] = len(self.vocab)
                self.decode_vocab[len(self.decode_vocab)] = new_vocab_token
            
            new_word_counts = Counter()
            for word, count in word_counts.items():
                new_word = word.replace(best_pair[0] + best_pair[1], new_vocab_token)
                new_word_counts[new_word]+= count
            word_counts = new_word_counts
            
            if len(self.vocab) >= self.vocab_size: #if vocabulary reaches the desired size, break
                break

    def _get_pairs(self, word_counts):
        """Finds the most frequent pairs in the words."""
        pairs = Counter()
        for word, count in word_counts.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
              pairs[(word[i], word[i+1])] += count
        return pairs
    
    def _encode_token(self, token):
        """Encodes a single token using BPE."""
        if token in self.vocab: #check first if the token is present in the vocabulary
            return [self.vocab[token]]
        
        encoded_token = []
        start = 0
        while start < len(token):
            best_match = ""
            for i in range(start + 1, len(token) + 1):
                sub = token[start:i]
                if sub in self.vocab and len(sub) > len(best_match):
                    best_match = sub
            
            if best_match: # if a match was found append
                encoded_token.append(self.vocab[best_match])
                start += len(best_match)
            else: # if not found, then break the word to chars and encode them, if they dont exists add <unk>
                if len(token[start:])> 0:
                  if token[start] in self.vocab:
                      encoded_token.append(self.vocab[token[start]])
                  elif "<unk>" in self.vocab:
                       encoded_token.append(self.vocab["<unk>"])
                  else: 
                      self.vocab["<unk>"] = len(self.vocab) # if <unk> does not exist add it to vocabulary
                      self.decode_vocab[len(self.decode_vocab)] = "<unk>"
                      encoded_token.append(self.vocab["<unk>"])
                start += 1
        return encoded_token

    def encode(self, text):
        return self.tokenize(text)

    def decode(self, tokens):
        decoded_text = []
        for token in tokens:
            if token in self.decode_vocab:
              decoded_text.append(self.decode_vocab[token])
            
        return ' '.join(decoded_text)


    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'level': self.level,
                'vocab': self.vocab,
                'decode_vocab': self.decode_vocab,
                 'vocab_size' : self.vocab_size,
                 'bpe_iterations' : self.bpe_iterations,
                 'custom_tokens' : self.custom_tokens
            }, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.level = data['level']
            self.vocab = data['vocab']
            self.decode_vocab = {int(k): v for k, v in data['decode_vocab'].items()}
            self.vocab_size = data['vocab_size']
            self.bpe_iterations = data['bpe_iterations']
            self.custom_tokens = data['custom_tokens']


# Test case with more complex text to show word splitting
if __name__ == '__main__':
    tokenizer = Tokenizer(level='sub_word', max_vocab_size=50000, bpe_iterations=10) # Lower vocab size and bpe iterations to better show the splits
    text = "This is a test, testing the subword tokenization. Tokenization of words with similar structure, like tokenizing and tokenized. Algorithm is fun and hiragana"
    tokenizer.tokenize("This is a test, testing the subword tokenization. Tokenization of words with similar structure, like tokenizing and tokenized")
    encoded_text = tokenizer.encode(text)
    print(f"Original text: {text}")
    print("\nEncoded text:")
    print(encoded_text)


    print("\nDecoded text (with original tokens):")
    decoded_text = tokenizer.decode(encoded_text)
    print(decoded_text)

    print("\nVocabulary:")
    for key, value in tokenizer.vocab.items():
        print(f"  {key}: {value}")

    print("\nDecode Vocabulary")
    for key, value in tokenizer.decode_vocab.items():
        print(f" {key}: {value}")