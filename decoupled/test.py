from datasets import load_dataset
from tokenizer import Tokenizer
import pickle
from tqdm import tqdm

tokenizer = Tokenizer(level='sub_word', max_vocab_size=1000000, bpe_iterations=10, custom_tokens=['<end>'])
ds = load_dataset("agentlans/high-quality-english-sentences")
text = ''
print(len(ds['train']))
# ds = ds['train']
# print(ds[0]['text'])
for i in tqdm(ds['train']):
    # print(i)
    # break
    text += i['text']
    text += '\n'
# tokenizer.tokenize(text)
print(len(tokenizer.vocab))
tokenizer.save("tokenizer.json")