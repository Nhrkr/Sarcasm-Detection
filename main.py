from models import lstm_model
from models import train_model
from preprocess import load_data
from preprocess import load_fast_text_embedding
from preprocess import get_embed_weights

max_len=50
embedding_dim=300
tokenizer, x_train, y_train, x_test, y_test,vocab_size= load_data('sarcasm_v2.csv')

embedding_index=load_fast_text_embedding('wiki-news-300d-1M-subword.vec')
embedding_matrix = get_embed_weights(embedding_index,tokenizer)
model=lstm_model(vocab_size,embedding_matrix)
train_model(model,x_train, y_train, x_test, y_test)




#half embeddingd
# Validation Loss:1.128701367992565 	Validation Accuracy:0.6840490793889286
# Validation Accuracy:68.40% (+/- 0.00%)
# Train Loss:1.1293133928731907 	Train Accuracy:68.40490797546013
# Test Loss:1.110706667958593 	Test Accuracy:71.16564420834641


