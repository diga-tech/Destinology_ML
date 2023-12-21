from __future__ import division  # Compatibility for division

import pandas as pd
import numpy as np
# import seaborn as sns

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
        epochs=10,
        validation_data=(x_val, y_val),
    )

    # Menyiapkan dataframe
    # place_df = place[['Place_Id', 'Place_Name', 'Category', 'Rating', 'Price']]
    # place_df.columns = ['id', 'place_name', 'category', 'rating', 'price']
    # df = rating.copy()

    # Mengambil sample user
    # user_id = df.User_Id.sample(1).iloc[0]
    # user_id = 50
    # place_visited_by_user = df[df.User_Id == user_id]

    # # Membuat data lokasi yang belum dikunjungi user
    # place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id']
    # place_not_visited = list(
    #     set(place_not_visited)
    #     .intersection(set(place_to_place_encoded.keys()))
    # )

    # place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    # user_encoder = user_to_user_encoded.get(user_id)
    # user_place_array = np.hstack(
    #     ([[user_encoder]] * len(place_not_visited), place_not_visited)
    # )

    # # Mengambil top 7 recommendation
    # ratings = model.predict(user_place_array).flatten()
    # top_ratings_indices = ratings.argsort()[-7:][::-1]
    # recommended_place_ids = [
    #     place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
    # ]

    # print('Daftar rekomendasi untuk: {}'.format('User ' + str(user_id)))
    # print('===' * 15, '\n')
    # print('----' * 15)
    # print('Tempat dengan rating wisata paling tinggi dari user')
    # print('----' * 15)

    # top_place_user = (
    #     place_visited_by_user.sort_values(
    #         by='Place_Ratings',
    #         ascending=False
    #     )
    #     .head(10)
    #     .Place_Id.values
    # )

    # place_df_rows = place_df[place_df['id'].isin(top_place_user)]
    # for row in place_df_rows.itertuples():
    #     print(row.place_name, ':', row.category)

    # print('')
    # print('----' * 15)
    # print('Top 7 place recommendation')
    # print('----' * 15)

    # recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
    # for row, i in zip(recommended_place.itertuples(), range(1, 9)):
    #     print(i, '.', row.place_name, '\n    ', row.category, ',', 'Harga Tiket Masuk ', row.price, ',',
    #           'Rating Wisata ', row.rating, '\n')

    # print('===' * 15)

    return model

if __name__ == "__main__":
    model=main()
    # model.save('satu_saved_model', save_format='tf')
    model_json = model.to_json()
    with open('my_recommender_model.json', 'w') as f:
        f.write(model_json)

    # Save weights
    model.save_weights('my_recommender_weights.h5')


