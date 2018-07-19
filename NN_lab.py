
# coding: utf-8

# <h1 align="center">TensorFlow Neural Network Lab</h1>

# <img src="image/notmnist.png">
# In this lab, you'll use all the tools you learned from *Introduction to TensorFlow* to label images of English letters! The data you are using, <a href="http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html">notMNIST</a>, consists of images of a letter from A to J in differents font.
# 
# The above images are a few examples of the data you'll be training on. After training the network, you will compare your prediction model against test data. Your goal, by the end of this lab, is to make predictions against that test set with at least an 80% accuracy. Let's jump in!

# To start this lab, you first need to import all the necessary modules. Run the code below. If it runs successfully, it will print "`All modules imported`".

# In[3]:


import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

print('All modules imported.')


# The notMNIST dataset is too large for many computers to handle.  It contains 500,000 images for just training.  You'll be using a subset of this data, 15,000 images for each label (A-J).

# In[4]:


def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

# Download the training and test dataset.
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

# Make sure the files aren't corrupted
assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',        'notMNIST_test.zip file is corrupted.  Remove the file and try again.'

# Wait until you see that all files have been downloaded.
print('All files downloaded.')


# In[5]:


def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    ダウンロードしてある zip ファイルを解凍し、 features と labels に分けて得る.
    
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar 
        # tqdm/tqdm 進捗バーを出すライブラリ https://github.com/tqdm/tqdm
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# Get the features and labels from the zip files
train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')

# データサイズを表示してみる. 1sample 当たりの train(画像)サイズ は 28x28=784 になるはず
print('features: train {}, labels {}'.format(train_features.shape, train_labels.shape))
print('test: train {}, labels {}'.format(test_features.shape, test_labels.shape))
print()
# 

# Limit the amount of data to work with
# 学習データが多すぎる場合、減らす
size_limit = 150000 
train_features, train_labels = resample(train_features, train_labels, n_samples=size_limit)

# Set flags for feature engineering.  This will prevent you from skipping an important step.
is_features_normal = False
is_labels_encod = False

# Wait until you see that all features and labels have been uncompressed.
print('All features and labels uncompressed.')


# <img src="image/mean_variance.png" style="height: 75%;width: 75%; position: relative; right: 5%">
# ## Problem 1
# The first problem involves normalizing the features for your training and test data.
# 
# Implement Min-Max scaling in the `normalize()` function to a range of `a=0.1` and `b=0.9`. After scaling, the values of the pixels in the input data should range from 0.1 to 0.9.
# 
# Since the raw notMNIST image data is in [grayscale](https://en.wikipedia.org/wiki/Grayscale), the current values range from a min of 0 to a max of 255.
# 
# Min-Max Scaling:
# $
# X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}
# $
# 
# *If you're having trouble solving problem 1, you can view the solution [here](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/solutions.ipynb).*

# In[7]:


# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # DONE: Implement Min-Max scaling for grayscale image data
    a = 0.1
    b = 0.9
    x = image_data
    min_of_x = np.min(x)
    max_of_x = np.max(x)
    return a + (x-min_of_x) * (b-a) / (max_of_x-min_of_x)

### DON'T MODIFY ANYTHING BELOW ###
# Test Cases
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
    [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     0.125098039216, 0.128235294118, 0.13137254902, 0.9],
    decimal=3)
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])),
    [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
     0.896862745098, 0.9])

if not is_features_normal:
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    is_features_normal = True

print('Tests Passed!')


# In[8]:


if not is_labels_encod:
    # Turn labels into numbers and apply One-Hot Encoding
    # One-Hot Encoding をラベルに適用
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True

# One-Hot Encoding のため、labels (10000,) なら (10000,10) のようにデータは 10倍になったはず
print('Encoded Labels: train {}, test {}'.format(train_labels.shape, test_labels.shape))
print()

print('Labels One-Hot Encoded')


# In[9]:


assert is_features_normal, 'You skipped the step to normalize the features'
assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'

# Get randomized datasets for training and validation
# 学習用データから、さらに学習用と検証用に分ける
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('features: train {}, labels {}'.format(train_features.shape, train_labels.shape))
print('valid: train {}, labels {}'.format(valid_features.shape, valid_labels.shape))
print()
# 7500 / 150000 = 0.05 = test_size

print('Training features and labels randomized and split.')


# In[11]:


# Save the data for easy access
# キャッシュとして、分けたデータを保存しておく
import gzip # サイズが 500MB 近いので圧縮したい

pickle_file = 'notMNIST.pickle.gz'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with gzip.open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')


# # Checkpoint
# All your progress is now saved to the pickle file.  If you need to leave and comeback to this lab, you no longer have to start from the beginning.  Just run the code block below and it will load all the data and modules required to proceed.

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')

import gc
print('collecting gc...')
gc.collect()
print('gc collected.')

# Load the modules
import pickle
import gzip # 圧縮されたキャッシュ用
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
pickle_file = 'notMNIST.pickle.gz'
with gzip.open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # Free up memory


print('Data and modules loaded.')

all_data = [train_features, train_labels, valid_features, valid_labels, test_features, test_labels]
shapes = [ x.shape for x in all_data ]
print('  data shapes: {}, {}, {}, {}, {}, {}'.format(*shapes))


# <img src="image/weight_biases.png" style="height: 60%;width: 60%; position: relative; right: 10%">
# ## Problem 2
# For the neural network to train on your data, you need the following <a href="https://www.tensorflow.org/resources/dims_types.html#data-types">float32</a> tensors:
#  - `features`
#   - Placeholder tensor for feature data (`train_features`/`valid_features`/`test_features`)
#  - `labels`
#   - Placeholder tensor for label data (`train_labels`/`valid_labels`/`test_labels`)
#  - `weights`
#   - Variable Tensor with random numbers from a truncated normal distribution.
#     - See <a href="https://www.tensorflow.org/api_docs/python/constant_op.html#truncated_normal">`tf.truncated_normal()` documentation</a> for help.
#  - `biases`
#   - Variable Tensor with all zeros.
#     - See <a href="https://www.tensorflow.org/api_docs/python/constant_op.html#zeros"> `tf.zeros()` documentation</a> for help.
# 
# *If you're having trouble solving problem 2, review "TensorFlow Linear Function" section of the class.  If that doesn't help, the solution for this problem is available [here](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/solutions.ipynb).*

# In[28]:


# Placeholder と変数の定義

features_count = 784   # = 28*28 (image size)
labels_count = 10      # 出力の分類数

# DONE: Set the features and labels tensors
# 初期値を入れたい変数はこれら
features = tf.placeholder(tf.float32, [None, features_count])
labels = tf.placeholder(tf.float32, [None, labels_count])

# DONE: Set the weights and biases tensors
# 求めたい変数はこれら
weights = tf.Variable(tf.truncated_normal([features_count, labels_count]))
biases = tf.Variable(tf.zeros([labels_count]))


# 以下、テストケース

### DON'T MODIFY ANYTHING BELOW ###

#Test Cases
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

assert features._shape == None or (    features._shape.dims[0].value is None and    features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels._shape  == None or (    labels._shape.dims[0].value is None and    labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
assert weights._variable._shape == (784, 10), 'The shape of weights is incorrect'
assert biases._variable._shape == (10), 'The shape of biases is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'

# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

# Linear Function WX + b
#logits = tf.matmul(features, weights) + biases
logits = features @ weights + biases

prediction = tf.nn.softmax(logits)

# Cross entropy
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)

# Training loss
loss = tf.reduce_mean(cross_entropy)

# Create an operation that initializes all variables
init = tf.global_variables_initializer()

# Test Cases
with tf.Session() as session:
    session.run(init)
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    session.run(loss, feed_dict=test_feed_dict)
    biases_data = session.run(biases)

assert not np.count_nonzero(biases_data), 'biases must be zeros'

print('Tests Passed!')


# In[29]:


# 精度計算用の定義

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print('Accuracy function created.')


# <img src="image/learn_rate_tune.png" style="height: 60%;width: 60%">
# ## Problem 3
# Below are 3 parameter configurations for training the neural network. In each configuration, one of the parameters has multiple options. For each configuration, choose the option that gives the best acccuracy.
# 
# Parameter configurations:
# 
# Configuration 1
# * **Epochs:** 1
# * **Batch Size:**
#   * 2000: ほぼ変わらず
#       ![image/plot/.png](./image/plot/epochs-1_batch_size-2000_learning_rate-0.01.png)
#       > Validation accuracy at 0.10040000081062317
#   * 1000: ほぼ変わらず
#       > Validation accuracy at 0.17573332786560059
#   * 500: ほぼ変わらず
#       > Validation accuracy at 0.10040000081062317
#   * 300: batch毎に徐々に精度アップ, loss減
#       ![image/plot/.png](./image/plot/epochs-1_batch_size-300_learning_rate-0.01.png)
#       > Validation accuracy at 0.352400004863739
#   * 50: batch毎に徐々に精度アップ, lossが減
#       ![image/plot/.png](./image/plot/epochs-1_batch_size-50_learning_rate-0.01.png)
#       > Validation accuracy at 0.6647999882698059
# * **Learning Rate:** 0.01
# 
# Configuration 2
# * **Epochs:** 1
# * **Batch Size:** 100
# * **Learning Rate:**
#   * 0.8
#     ![image/plot/.png](./image/plot/epochs-1_batch_size-100_learning_rate-0.8.png)
#     > Validation accuracy at 0.10040000081062317
#   * 0.5
#     ![image/plot/.png](./image/plot/epochs-1_batch_size-100_learning_rate-0.5.png)
#     > Validation accuracy at 0.7370666861534119
#   * 0.1
#     ![image/plot/.png](./image/plot/epochs-1_batch_size-100_learning_rate-0.05.png)
#     
#     > Validation accuracy at 0.7482666373252869
#   * 0.05
#     ![image/plot/.png](./image/plot/epochs-1_batch_size-100_learning_rate-0.05.png)
#   
#     > Validation accuracy at 0.7289333343505859
#   * 0.01
#     ![image/plot/.png](./image/plot/epochs-1_batch_size-100_learning_rate-0.01.png)
#     
#     > Validation accuracy at 0.5685333609580994
#     
# Configuration 3
# * **Epochs:**
#   * 1
#     ![image/plot/.png](./image/plot/epochs-1_batch_size-100_learning_rate-0.2.png)
#     > Validation accuracy at 0.7574666738510132
#   * 2
#     ![image/plot/.png](./image/plot/epochs-2_batch_size-100_learning_rate-0.2.png)  
#     > Validation accuracy at 0.7710666656494141
#   * 3
#     ![image/plot/.png](./image/plot/epochs-3_batch_size-100_learning_rate-0.2.png)
#     > Validation accuracy at 0.7793333530426025
#   * 4
#     ![image/plot/.png](./image/plot/epochs-4_batch_size-100_learning_rate-0.2.png)
#     > Validation accuracy at 0.7581333518028259
#   * 5
#     ![image/plot/.png](./image/plot/epochs-5_batch_size-100_learning_rate-0.2.png)
#     > Validation accuracy at 0.7879999876022339
# * **Batch Size:** 100
# * **Learning Rate:** 0.2
# 
# The code will print out a Loss and Accuracy graph, so you can see how well the neural network performed.
# 
# *If you're having trouble solving problem 3, you can view the solution [here](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/solutions.ipynb).*

# In[30]:


# DONE: Find the best parameters for each configuration
epochs = 2
batch_size = 100
learning_rate = 0.2


### DON'T MODIFY ANYTHING BELOW ###
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    # 繰り返し学習させる
    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # mini-batch のサイズごとに実行
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            # mini-batch を作成
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})
            
            # 特定バッチ回数毎に精度をログる
            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # epoch 毎に精度を計算する
        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()

# 画像を保存
filename = './image/plot/epochs-{}_batch_size-{}_learning_rate-{}.png'.format(epochs, batch_size, learning_rate)
print('Save the plot image to', filename)
plt.savefig(filename)

plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))


# ## Test
# Set the epochs, batch_size, and learning_rate with the best learning parameters you discovered in problem 3.  You're going to test your model against your hold out dataset/testing data.  This will give you a good indicator of how well the model will do in the real world.  You should have a test accuracy of at least 80%.
# 
# 上記を参考に 80% 以上の精度を出してみる
# 
# > Nice Job! Test Accuracy is 0.8561000227928162
# 
# loss計算やログをとらなければ早い？

# In[35]:


# DONE: Set the epochs, batch_size, and learning_rate with the best parameters from problem 3
epochs = 5
batch_size = 50
learning_rate = 0.01



### DON'T MODIFY ANYTHING BELOW ###
# The accuracy measured against the test set
test_accuracy = 0.0

with tf.Session() as session:
    
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer
            _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

        print('test_accuracy at {}'.format(test_accuracy))

assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))


# # Multiple layers
# Good job!  You built a one layer TensorFlow network!  However, you want to build more than one layer.  This is deep learning after all!  In the next section, you will start to satisfy your need for more layers.
