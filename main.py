import sys
from transformers import LlamaTokenizerFast, CLIPTokenizer

# ✅ Get trigger word from command-line argument
if len(sys.argv) < 2:
    raise ValueError("Please provide a trigger word as the first argument.")
trigger_word = sys.argv[1]

# ✅ Load HunyuanVideo model from Hugging Face
model_name = "hunyuanvideo-community/HunyuanVideo"
tokenizer = LlamaTokenizerFast.from_pretrained(
    model_name, subfolder="tokenizer"
)
#tokenizer_2 = CLIPTokenizer.from_pretrained(
#    model_name, subfolder="tokenizer_2"
#)

tokens = tokenizer.tokenize(trigger_word)
#tokens2 = tokenizer_2.tokenize(trigger_word)
print(tokens)
#print(tokens2)

