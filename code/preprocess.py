from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import re 

WINDOW_SIZE = 64
VOCAB_SIZE = 10000
PADDING_INDEX = 4
MIN_FREQUENCY = 2


'''
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
        else: 
          non_lines.append(str(line_text))
  return char, lines
'''


def read_friends_data(file_name):
  """
  DO NOT CHANGE
  Load text data from file
  :param file_name:  string, name of data file
  :return: list of sentences, each a list of words split on whitespace
  """
  lines = []
  with open(file_name, 'rt', encoding='latin',newline="") as data_file:
    for line in data_file:
      line = re.sub("[\(\[].*?[\)\]]", "", line)
      if ":" in line: 
        colon_indices = [i for i in range(len(line)) if line.startswith(":", i)]
        space_indices = [i for i in range(len(line)) if line.startswith(" ", i)]
        if len(colon_indices) == 1:
          lines.append(line)
        else: 
          for i in range(len(colon_indices)): 
            character_name_index_last = colon_indices[i] - 1
            space_indices_filter = [i for i in space_indices if i < character_name_index_last]
            if len(space_indices_filter) != 0:
              character_name_index_start = min(space_indices_filter, key=lambda x:abs(x-character_name_index_last))
            else: 
              character_name_index_start = 0
            character_name = line[character_name_index_start: character_name_index_last+1]
            if (character_name.isupper()):
              if i != len(colon_indices) - 1: 
                next_character_name_index_last = colon_indices[i+1] - 1
                space_indices_filter = [i for i in space_indices if i < next_character_name_index_last]
                next_character_name_index_start = min(space_indices_filter, key=lambda x:abs(x-next_character_name_index_last))
                next_character_name = line[next_character_name_index_start: next_character_name_index_last+1]
                if (next_character_name.isupper()): 
                  lines.append(line[character_name_index_start: next_character_name_index_start])
                else: 
                  lines.append(line[character_name_index_start:])
              else: 
                lines.append(line[character_name_index_start:])
            else: 
              lines.append(line)

    filtered_lines = []
    char = []
    for line in lines: 
      if "THE ONE" not in line and "Written by:" not in line:
        line_text = line.rstrip('\n').split(": ", 1)
        if len(line_text) == 2: 
          char.append(line_text[0])
          filtered_lines.append(line_text[1])
    return char, filtered_lines


def read_pretrain_data(file_name):
  lines = []
  with open(file_name, 'rt', encoding='latin',newline="") as data_file:
    for line in data_file:
      line = line.encode("ascii","ignore").decode()
      if line != " \n" and "= " not in line:
        for sentence in line.rstrip(' \n').split("."):
          lines.append(sentence + ".")
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


