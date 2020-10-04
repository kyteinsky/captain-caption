from helper import *
from config import hyperparams
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from time import sleep
import re
import numpy as np
import time
import json
from glob import glob
import pickle
from tqdm import tqdm, trange
import wandb
# import click
import argparse
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

npy_dir, EPOCHS, sample_size, BATCH_SIZE, BUFFER_SIZE, embedding_dim, units, top_k, features_shape, attention_features_shape, cpt, wb, npy = hyperparams()

# print(BATCH_SIZE)
# @click.command()
# @click.option('--batch_size', default=BATCH_SIZE)
# @click.option('--buffer_size', default=BUFFER_SIZE)
# @click.option('--embed_dim', default=embedding_dim)
# @click.option('--epochs', default=EPOCHS)
# @click.option('--unitss', default=units)
# def hello(batch_size: int, buffer_size: int, embed_dim: int, epochs: int, unitss: int):
#     BATCH_SIZE = batch_size
#     BUFFER_SIZE = buffer_size
#     embedding_dim = embed_dim
#     EPOCHS = epochs
#     units = unitss
#     method()
#     print('here')



ap = argparse.ArgumentParser()
ap.add_argument('--train_path', type=str, default='train2014/')
ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
ap.add_argument('--buffer_size', type=int, default=BUFFER_SIZE)
ap.add_argument('--epochs', type=int, default=EPOCHS)
ap.add_argument('--ckpt', default="./checkpoints")
args = vars(ap.parse_args())

checkpoint_path = os.path.join(args['ckpt'], 'train/')

BATCH_SIZE, BUFFER_SIZE, EPOCHS = args['batch_size'], args['buffer_size'], args['epochs']

vocab_size = top_k + 1
if wb:
    config={'batch_size' : BATCH_SIZE,
            'epochs' : EPOCHS,
            'buffer_size' : BUFFER_SIZE
            }
    wandb.init(project="azure-captioning", sync_tensorboard=True, config=config)



annotation_file = 'annotations/captions_train2014.json'
PATH = args['train_path']

########################################------1------# pre-steps
# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

train_captions = train_captions[:sample_size]
img_name_vector = img_name_vector[:sample_size]

del all_captions
del all_img_name_vector

print("DON'T FORGET TO MOUNT NPY FILES DRIVE!!")
print('1')
########################################------2------# pretrained model - Inception V3
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


print('2')
########################################------3------# caching features
# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(128)

if npy:
    i = 0
    for img, path in tqdm(image_dataset):
        i += 1
        if i % 500 == 0: sleep(20)
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            path_of_feature = os.path.join(npy_dir, os.path.basename(path_of_feature))
            if not os.path.isfile(path_of_feature+'.npy'):
                np.save(path_of_feature, bf.numpy())

    del i
    exit()


# for _,i in enumerate(img_name_vector):
#     if _ == 1: break
#     print(np.array(img_name_vector))
# if np.array_equal(a, np.array(img_name_vector)[1]): print('ok')
# exit()




########################################------4------# preprocessing captions
if not os.path.isfile('tokenizer.json'):
# # Choose the top 5000 words from the vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # saving tokenizer
    # with open('tokenizer.pickle', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tokenizer_json = tokenizer.to_json()

    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

else: # load the tokenizer file
    with open('tokenizer.json') as f:
        datax = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(datax)
        del datax


# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)


print('3')
########################################------5------# split data
# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)



########################################------6------# create tf.dataset
dataset = tfdataset(img_name_train, cap_train)


########################################------7------# model
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


########################################------8------# checkpoint
start_epoch = 0
if cpt:
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)







########################################------8------# train step
@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss





########################################------8------# TRAIN #-----------------#################################################
num_steps = len(img_name_train) // BATCH_SIZE
print('start training ..')
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in tqdm(enumerate(dataset), ascii=True):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if wb and batch % 10 == 0:
            wandb.log({'loss':batch_loss.numpy() / int(target.shape[1])})

        if (batch+1) % 100 == 0:
            print ('Epoch {}/Epochs {} Batch {} Loss {:.4f}'.format(
              epoch + 1, EPOCHS, batch, batch_loss.numpy() / int(target.shape[1])))
            sleep(20)

    if epoch % 5 == 0 and cpt:
        ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))

    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


