from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

WINDOW_SIZE = 64
VOCAB_SIZE = 10000
PADDING_INDEX = 4
MIN_FREQUENCY = 2

def read_friends_data(file_name):
  """
  DO NOT CHANGE
  Load text data from file
  :param file_name:  string, name of data file
  :return: list of sentences, each a list of words split on whitespace
  """
  lines = []
  char = []
  with open(file_name, 'rt', encoding='latin',newline="") as data_file:
    for line in data_file:
      if line[0] != '[' and line[0] != '(' and "THE ONE" not in line and "Written by:" not in line:
        line_text = line.rstrip('\n').split(": ", 1)
        if len(line_text) == 2:
          char.append(line_text[0])
          lines.append(line_text[1])
  return char, lines

def read_pretrain_data(file_name):
  lines = []
  with open(file_name, 'rt', encoding='latin',newline="") as data_file:
    for line in data_file:
      line = line.encode("ascii","ignore").decode()
      if line != " \n" and "= " not in line:
        for sentence in line.rstrip(' \n').split("."):
          lines.append(sentence + ".")
  print(len(lines))
  return lines


def get_data():
  file_name = ["../data/friends_transcript.txt", "../data/wikitext-2-raw/wiki.train.raw"]
  pretrain_text = read_pretrain_data(file_name[1])
  chars, lines = read_friends_data(file_name[0])

  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  tokenizer.enable_padding(direction="right", pad_id=4,
                                  pad_token='[PAD]',
                                  length=WINDOW_SIZE+1)
  trainer = BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]", "[DELIM]", "[PAD]"], 
    vocab_size=VOCAB_SIZE, min_frequency = 5)
  tokenizer.pre_tokenizer = Whitespace()
  tokenizer.train(file_name, trainer)
  tokenizer.post_processor = TemplateProcessing(
      single="[BOS] $A [EOS]",
      pair="[BOS] $A [DELIM] $B:1 [EOS]:1",
      special_tokens=[
          ("[BOS]", tokenizer.token_to_id("[BOS]")),
          ("[EOS]", tokenizer.token_to_id("[EOS]")),
          ("[DELIM]", tokenizer.token_to_id("[DELIM]")),
      ],
  )

  friends_data = []
  for i in range(len(lines)): 
    output = tokenizer.encode(chars[i], lines[i])
    if len(output.ids) == WINDOW_SIZE+1:
      friends_data.append(output.ids)

  pretrain_data = tokenizer.encode_batch(pretrain_text)
  pretrain_data = [x.ids for x in pretrain_data if len(x)==WINDOW_SIZE+1]
  

  return tokenizer, friends_data, pretrain_data
    




