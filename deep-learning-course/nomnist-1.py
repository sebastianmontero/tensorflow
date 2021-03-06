from __future__ import print_function
import imageio
import matplotlib as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from numpy import dtype


url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = './data' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename
  
def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (imageio.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

def make_arrays(nb_rows, img_size, num_classes, convolution=False):
  if nb_rows:
    shape = (nb_rows, img_size, img_size, 1) if convolution else (nb_rows, img_size ** 2) 
    dataset = np.ndarray(shape, dtype=np.float32)
    labels = np.ndarray((nb_rows, num_classes), dtype=np.float32)
  else:
    dataset, labels = None, None
  return dataset, labels
  
def reshape_image(image_array, num_images, image_size, convolution=False):
    shape = [num_images, image_size, image_size, 1] if convolution else [num_images, -1] 
    return np.reshape(image_array, shape)

def reshape_label(labels, num_classes):
    return np.eye(10, dtype=np.float32)[labels]

def merge_datasets(pickle_files, train_size, valid_size=0, convolution=False):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size, num_classes, convolution)
  train_dataset, train_labels = make_arrays(train_size, image_size, num_classes, convolution)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
   
  hot_classes = np.eye(num_classes, dtype=np.float32) 
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :] = reshape_image(valid_letter, vsize_per_class, image_size, convolution)
          valid_labels[start_v:end_v] = hot_classes[label]
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :] = reshape_image(train_letter, tsize_per_class, image_size, convolution)
        train_labels[start_t:end_t] = hot_classes[label]
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
  
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
    
def maybe_pickle_final_dataset(train_datasets, test_datasets, train_size, valid_size, test_size, convolution=False,  force=False):
    
    file_name = 'notMNIST-conv.pickle' if convolution else 'notMNIST.pickle'
    
    pickle_file = os.path.join(data_root, file_name)

    if os.path.exists(pickle_file) and not force:
        print('Final dataset pickle present, loading from pickle')
        try:
            with open(pickle_file, 'rb') as f:
                dataset = pickle.load(f)
            
            train_dataset, train_labels = dataset['train_dataset'], dataset['train_labels'] 
            valid_dataset, valid_labels = dataset['valid_dataset'], dataset['valid_labels']
            test_dataset, test_labels = dataset['test_dataset'], dataset['test_labels']
        except Exception as e:
            print('Unable to read data to', pickle_file, ':', e)
            raise
    else:
        print('Final dataset not present, merging datasets')
        valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
            train_datasets, train_size, valid_size, True)
        _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size, convolution=convolution)

        train_dataset, train_labels = randomize(train_dataset, train_labels)
        test_dataset, test_labels = randomize(test_dataset, test_labels)
        valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
        try:
            f = open(pickle_file, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
                }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    
    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    statinfo  = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)
    
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def train_linear(train_dataset, train_labels, valid_dataset, valid_labels):
    
    model = LogisticRegression()
    model.fit(train_dataset, train_labels)
    print('Score on valid dataset:', model.score(valid_dataset, valid_labels))
            
train_size = 200000
valid_size = 10000
test_size = 10000

num_classes = 10
np.random.seed(133)
image_size  = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = maybe_pickle_final_dataset(train_datasets, test_datasets, train_size, valid_size, test_size, True, True)
print(train_labels)
#train_linear(train_dataset, train_labels, valid_dataset, valid_labels)