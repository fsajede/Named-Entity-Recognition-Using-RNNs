import A2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Dense, TimeDistributed
from tensorflow.keras.initializers import Constant

sent,pos,ner = A2.read_data()
len(sent)
sent[0]
pos[0]
ner[0]
wordmap, reverse_wordmap = A2.make_map(sent)
vocab_len = len(wordmap)
# Use the islice function from itertools to get the first five items
from itertools import islice


# Use islice to get the first two key-value pairs
result_dict = dict(islice(wordmap.items(), 5))

print(result_dict)
vocab_len
reverse_wordmap[2]
tagmap, reverse_tagmap = A2.make_map(ner)
tag_len = len(tagmap)
tag_len
tagmap
reverse_tagmap[3]
# creating training and test sets
tr_sent = sent[:40000]
tr_tag = ner[:40000]
te_sent = sent[40000:]
te_tag = ner[40000:]
# preparing training data
maxlen = 30
v_tr_sent = A2.vectorize_sent(tr_sent,wordmap,maxlen)
v_tr_target =A2. vectorize_sent(tr_tag,tagmap,maxlen)
len(v_tr_sent)
v_tr_sent
v_tr_sent[0]
len(v_tr_target)
v_tr_target.shape
v_tr_target
v_tr_target[0]
# One-Hot Targets
v_tr_target = to_categorical(v_tr_target,num_classes=tag_len)
v_tr_target.shape
v_tr_target[0]
embedding_matrix =A2. get_embedding_matrix(wordmap)
embedding_matrix.shape
# Extract embed_len from the shape of the embedding_matrix
embed_len = embedding_matrix.shape[1]
embed_len
#### First Model
#Building RNN Model
m = Sequential()

#Learning Embeddings
m.add(Embedding(vocab_len, embed_len,embeddings_initializer=Constant(embedding_matrix)))

#Add RNN Layer
m.add(SimpleRNN(10, return_sequences=True, activation="relu"))


#Time Distributed Layer
m.add(TimeDistributed(Dense(50,activation="relu")))

#Output Layer
m.add(Dense(tag_len,activation="softmax"))
#Training the Model
m.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
m.fit(v_tr_sent, v_tr_target, batch_size=100, epochs=2, verbose=1)
#Making Predictions
te_predicted = A2.tag_sentences(te_sent, m, wordmap, reverse_tagmap, maxlen)
entity_types = ['geo', 'gpe', 'per', 'org', 'tim', 'art', 'nat', 'eve']
#Evaluation
A2.evaluate_ie(te_sent, te_tag, te_predicted, entity_types,"eval_model_1.txt")
#### Second Model
#Building RNN Model
m_2 = Sequential()

#Learning Embeddings
m_2.add(Embedding(vocab_len, embed_len,embeddings_initializer=Constant(embedding_matrix)))

#Add RNN Layer
m_2.add(Bidirectional(SimpleRNN(10, return_sequences=True, activation="relu")))

#Time Distributed Layer
m_2.add(TimeDistributed(Dense(50,activation="relu")))

#Output Layer
m_2.add(Dense(tag_len,activation="softmax"))
#Training the Model
m_2.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
m_2.fit(v_tr_sent, v_tr_target, batch_size=100, epochs=2, verbose=1)
#Making Predictions
te_predicted = A2.tag_sentences(te_sent, m_2, wordmap, reverse_tagmap, maxlen)
#Evaluation
A2.evaluate_ie(te_sent, te_tag, te_predicted, entity_types,"eval_model_2.txt")

#### Task3
# Writing sentences and NER tags to a file
with open("example_sentences.txt", "w") as file:
    sentences_and_tags = [
        ("The company is located in Paris, France.", ["O","O","O","O","O","B-geo","B-geo"]),
        ( "Apple Inc. was founded by Steve Jobs in Cupertino." ,["org"," O", "O", "O", "O", "B-per", "I-per"," O", "B-geo"]),
        ( "Jane Smith, a renowned scientist, works at NASA." ,["B-per", "I-per", "O"," O", "O", "O", "O", "B-org"]),
        ( "The Amazon River is the second-longest river in the world.",  ["O","B-geo", "I-geo", "O", "O", "O", "O", "O", "O", "O"]),
        ( "Harry Potter is a popular book series.", ["B-per", "I-per", "O", "O", "O", "O","O"]),
        ('David and Sarah got married ',["B-per","O","B-per","O","O"]),
        ('Jack sent email to Stella ',["B-per","O","O","O","B-per"]),
        ("Sarah found a good result last Monday",['B-per', 'O','O','O','O','O','B-tim'] ),
        
    ]
    for sentence, tags in sentences_and_tags:
        file.write(f"{sentence}\t{' '.join(tags)}\n")

# Reading sentences and NER tags from the file
my_test_sent = []
ner_tag_name = []

with open("example_sentences.txt", "r") as file:
    for line in file:
        sentence, tags = line.strip().split("\t")
        my_test_sent.append(sentence.split())
        ner_tag_name.append(tags.split())

# Print the result
print("my_test_sent:", my_test_sent)
print("ner_tag_name:", ner_tag_name)
wordmap_test, reverse_wordmap_test = A2.make_map(my_test_sent)
tagmap_test, reverse_tagmap_test = A2.make_map(ner_tag_name)
#Making Predictions
te_predicted_1 = A2.tag_sentences(my_test_sent, m_2, wordmap_test, reverse_tagmap_test, maxlen)
#Evaluation
A2.evaluate_ie(my_test_sent, ner_tag_name, te_predicted_1, entity_types,"eval_task_3.txt")