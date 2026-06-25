from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. Initialize a new, empty tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# 2. Create a trainer
# You can set the vocab_size you want
trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "[INST]", "[/INST]"], vocab_size=50200)

# 3. Train the tokenizer on your data file
# This might take a few minutes on a large file.
files = ["/Users/sameerhumagain/DeepLearning/sample-data/story.txt"]
tokenizer.train(files, trainer)

# 4. Save the new tokenizer
tokenizer.save("story-tokenizer.json")