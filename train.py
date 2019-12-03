import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras import Model
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Train model started')

assert tf.__version__ == '2.0.0-rc1'
print(tf.__version__)

# the dataset is cleaned already so we don't to work on it
data_set = pd.read_csv('data/ratings.csv')
train_set, test_set = train_test_split(data_set, test_size=0.3)
print(train_set.describe)

# needs for embedding layers
num_books = len(data_set.book_id.unique())
num_users = len(data_set.user_id.unique())

print('num_books:', num_books)
print('num_users:', num_users)

# creating book embedding path FIRST INPUT
input_books = Input(shape=(1,))
model_books = Embedding(num_books+1, 5)(input_books)
model_books = Flatten()(model_books)
model_books = Model(inputs=input_books, outputs=model_books)

# creating user embedding path SECOND INPUT
input_users = Input(shape=(1,))
model_users = Embedding(num_users+1, 5)(input_users)
model_users = Flatten()(model_users)
model_users = Model(inputs=input_users, outputs=model_users)

output_combined = Concatenate()([model_books.output, model_users.output])

#  use more dense layers
output_combined = Dense(128, activation='relu')(output_combined)
output_combined = Dense(64, activation='relu')(output_combined)
output_combined = Dense(1)(output_combined)

model = Model([input_users, input_books], output_combined)

# compile the model
model.compile('adam', loss='mse')

# Train  model
train_history = model.fit(
    [train_set.user_id, train_set.book_id],
    train_set.rating,
    batch_size=32,
    epochs=5,
    verbose=1
)

# show training results
plt.plot(train_history.history['loss'], label='loss')
plt.legend()
plt.show()

# evaluate model and show results
results = model.evaluate([test_set.user_id, test_set.book_id], test_set.rating)
print('evaluate results:', results)

