
from itertools import chain

# Let's define our contexts and special tokens
persona = [["i", "like", "playing", "football", "."],
           ["i", "am", "from", "NYC", "."]]
history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]
reply = ["great", "to", "hear"]
bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"

def build_inputs(persona, history, reply):
    # words is the embedding of just the words segmented by tokens
    # segments is an embedding indicating the speaker of each word
    # position is the literal, numerical position of each word
    # sequence is list of lists containing the sequences


    # Build our sequence by adding delimiters and concatenating
    sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]
    sequence = [sequence[0]] + [ [speaker2 if (len(sequence)-i) % 2 else speaker1] + s
                                for i, s in enumerate(sequence[1:])]
    # Build our word, segments and position inputs from the sequence
    words = list(chain(*sequence))                          # word tokens
    segments = [speaker2 if i % 2 else speaker1             # segment tokens
                for i, s in enumerate(sequence) for _ in s]
    position = list(range(len(words)))                      # position tokens
    return words, segments, position, sequence

words, segments, position, sequence = build_inputs(persona, history, reply)

print(segments)
print(position)
print(sequence)
# >>> print(sequence)  # Our inputs looks like this:
# [['<bos>', 'i', 'like', 'playing', 'football', '.', 'i', 'am', 'from', 'NYC', '.'],
#  ['<speaker1>', 'hello', 'how', 'are', 'you', '?'],
#  ['<speaker2>', 'i', 'am', 'fine', 'thanks', '.'],
#  ['<speaker1>', 'great', 'to', 'hear', '<eos>']]

# Tokenize words and segments embeddings:
#words = tokenizer.convert_tokens_to_ids(words)
#segments = tokenizer.convert_tokens_to_ids(segments)