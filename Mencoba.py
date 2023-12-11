from __future__ import division  # Compatibility for division

import pandas as pd
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import any additional libraries or callbacks needed

class RecommenderNet(tf.keras.Model):
    """
    Class for creating a recommender system using a neural network.
    """

    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        """
        Initialize the recommender system model.
        """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.places_embedding = layers.Embedding(
            num_places,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.places_bias = layers.Embedding(num_places, 1)

    def call(self, inputs):
        """
        Perform forward pass for the model.
        """
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        places_vector = self.places_embedding(inputs[:, 1])
        places_bias = self.places_bias(inputs[:, 1])

        dot_user_places = tf.tensordot(user_vector, places_vector, 2)

        x = dot_user_places + user_bias + places_bias

        return tf.nn.sigmoid(x)

def dict_encoder(col, data):
    """
    Encode categorical column values to numerical values.
    """
    if data is None:
        data = pd.DataFrame()
    unique_val = data[col].unique().tolist()
    val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
    return val_to_val_encoded, val_encoded_to_val

# Add the definition or import of 'myCallback' if needed

def main():
    """
    Main function to run the recommender system.
    """
    rating = pd.read_csv(r'./dataset/tourism_rating.csv')
    place = pd.read_csv(r'./dataset/tourism_with_id.csv')
    user = pd.read_csv(r'./dataset/user.csv')

    # Rest of the code remains unchanged
    place = place.drop(['Unnamed: 11', 'Unnamed: 12', 'Time_Minutes'], axis=1)
    rating = pd.merge(rating, place[['Place_Id']], how='right', on='Place_Id')
    user = pd.merge(user, rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')
    df = rating.copy()
    user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id', data=user)
    df['user'] = df['User_Id'].map(user_to_user_encoded)

    # Encoding Place_Id
    place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id', data=place)

    # Mapping Place_Id ke dataframe place
    df['place'] = df['Place_Id'].map(place_to_place_encoded)

    num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)

    # Mengubah rating menjadi nilai float
    df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)

    # Mendapatkan nilai minimum dan maksimum rating
    min_rating, max_rating = min(df['Place_Ratings']), max(df['Place_Ratings'])

    # Membuat variabel x untuk mencocokkan data user dan place menjadi satu value
    x = df[['user', 'place']].values

    # Membuat variabel y untuk membuat rating dari hasil
    y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    # Membagi menjadi 80% data train dan 20% data validasi
    train_indices = int(0.8 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )

    model = RecommenderNet(num_users, num_place, 50)  # inisialisasi model

    # model compile
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=100,
        validation_data=(x_val, y_val),
    )

    return model

if __name__ == "__main__":
    model=main()
    model.save('saved_model', save_format='tf')


