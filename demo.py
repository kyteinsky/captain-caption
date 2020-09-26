import cv2
import os
import numpy as np
from config import hyperparams
from helper import RNN_Decoder, CNN_Encoder, load_image
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from opencv_text import OpencvText
import pickle, io, json


npy_dir, EPOCHS, sample_size, BATCH_SIZE, BUFFER_SIZE, embedding_dim, units, top_k, features_shape, attention_features_shape, cpt, wb, npy = hyperparams()
max_length = 52
checkpoint_path = './checkpoints/train/'
vocab_size = top_k + 1
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
ckpt.restore(ckpt_manager.latest_checkpoint)

# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
#                                                   oov_token="<unk>",
#                                                   filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
# loading tokenizer
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)


def image_filenames(folder):
    img_files = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img_files.append(os.path.join(folder,filename))
    return img_files


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def save_attention_plot(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join('./test_images/output', 'attention_'+os.path.basename(image)))


if __name__ == '__main__':
    av_time = 0.0
    n = 0
    op = OpencvText(folder='./test_images')
    for im in image_filenames('./test_images/input'):
        n += 1
        st = time()
        result, attention_plot = evaluate(im)
        av_time += time()-st
        resultx = ' '.join(result)
        # print ('Prediction Caption:', resultx)
        op.run_image(im, resultx)
        break
        save_attention_plot(im, result, attention_plot)
    print('Time taken for each inference (avg): %.2f s' %(av_time/n))




