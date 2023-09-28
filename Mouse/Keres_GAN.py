import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import csv_reader

# Define the dimensions
input_dim = 2  # Size of the input point [x, y]
output_dim = 2  # Size of the generated points [x, y]
num_points = 1000

# Generator
def build_generator():
    input_point = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(input_point)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    generated_points = Dense(num_points * output_dim, activation='tanh')(x)
    generated_points = tf.reshape(generated_points, (-1, num_points, output_dim))
    return Model(input_point, generated_points)

# Discriminator
def build_discriminator():
    input_real_point = Input(shape=(input_dim,))
    input_generated_points = Input(shape=(num_points, output_dim))
    
    x = concatenate([input_real_point, tf.reshape(input_generated_points, (-1, num_points*output_dim))])
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    validity = Dense(1, activation='sigmoid')(x)
    return Model([input_real_point, input_generated_points], validity)

# Define the cGAN model
def build_cgan(generator, discriminator):
    input_point = Input(shape=(input_dim,))
    generated_points = generator(input_point)
    validity = discriminator([input_point, generated_points])
    return Model(input_point, [validity, generated_points])

# Define optimizer and loss function
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

generator = build_generator()

discriminator.trainable = False

cgan = build_cgan(generator, discriminator)
cgan.compile(loss=['binary_crossentropy', 'mean_squared_error'], optimizer=optimizer)

# Load your training data and labels here
d=csv_reader.data_array   #próbáljuk ki a csúnyábbik beolvasással, ahol még nem tuple-k

def create_labels(dataset):
    labels=[]
    for i in dataset:
        labels.append(i[0])
    return labels

labels=create_labels(d)
labels=np.array(labels)

def create_data(dataset):
    data=[]
    for j in dataset:
        d=[]
        for i in range(2,len(j)):
            d.append(j[i])
        data.append(d)
    return data

data=create_data(csv_reader.data)

data = np.asarray(data).astype(np.float32)
# data and labels should be numpy arrays

# Train the cGAN
batch_size = 1000
epochs = 1000
i=0
for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, data.shape[0], batch_size)
    real_points = data[idx]
    real_labels = np.ones((batch_size, 1))
    generated_points = generator.predict(real_points)
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch([real_points, generated_points], real_labels)
    d_loss_fake = discriminator.train_on_batch([real_points, generated_points], fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    g_loss = cgan.train_on_batch(real_points, [real_labels, real_points])

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss[0]}")
    i+=1