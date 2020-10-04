
def hyperparams():
    npy_dir = './npy'
    EPOCHS = 5
    sample_size = 414113#30000

    BATCH_SIZE = 512
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    top_k = 20000

    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64


    cpt = True
    wb = False
    npy = True

    return npy_dir, EPOCHS, sample_size, BATCH_SIZE, BUFFER_SIZE, embedding_dim, units, top_k, features_shape, attention_features_shape, cpt, wb, npy
