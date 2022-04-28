from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing



def read_data(file_name):
	"""
	DO NOT CHANGE
  Load text data from file
	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line)
	return text

file_name = "../data/office_transcript.csv"
data = read_data(file_name)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.enable_padding(direction="right", pad_id=0,
                                 pad_type_id=0, pad_token='[PAD]',
                                 length=128)

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train([file_name], trainer)
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

all_tokens = []
all_ids = []
token_ids = []
for line in data: 
    output = tokenizer.encode(line)
    ids = output.ids
    tokens = output.tokens
    all_tokens.append(tokens)
    all_ids.append(ids)
    tokens.insert(0,tokens[1])
    ids.insert(0, ids[1])
    tokens[1:] = ids[1:]
    token_id = tokens
    token_ids.append(token_id)

print(all_tokiens)
print(all_ids)
print(token_ids)    

    
    




