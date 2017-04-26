**(insert cover image)**

# Introduction
In my last post [Sparse Matrices For Efficient Machine Learning](insert link), I showcased methods and a workflow for converting a matrix with lots of zero values into a sparse matrix with Scipy. This did two things:

 1. compressed the in-memory footprint of the data matrix 
 2. sped up many machine learning routines

I ended that post with a challenge: **if the original data matrix won’t fit into memory in the first place, can you think of a way to convert it to a sparse matrix anyway?**

To solve this problem we need to think bigger. We need to step away from in-memory techniques and instead delve into on-disk methods.

In this vein, allow me to introduce Hierarchical Data Format version 5 (HDF5), an extremely powerful tool rife with capabilities. As best summarized in [Python and HDF5](http://shop.oreilly.com/product/0636920030249.do) 
> "HDF5 is just about perfect if you make minimal use of relational features and have a need for very high performance, partial I/O, hierarchical organization, and arbitrary
metadata".

While I will focus primarily on its on-the-fly compression capabilities and mention its partial I/O capabilities, I highly recommend *Python and HDF5* and the links provided in the *Additional Resources" section for those want to dig deep. 

Before digging in, allow me to introduce the dataset we will be working with.

# Dataset: Dota 2

**Insert Dota2 Image**

Dota 2 is a popular computer game published by Valve Corporation. Two teams consisting of five players are forged from among 113 heros, each with unique strengths and weaknesses. Players and teams gain experience and items during the course of the game, which ends when one team destroys the "Ancient", a structure in the opposing team's base.

I chose the [Dota2 Games Results Data Set](https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results#) for two reasons:

1. it is real-world data

2) many people are interested in gaming

The dataset captures information for all games played in a space of 2 hours on the 13th of August, 2016. Specifically, the dataset consists of a target variable and 116 features. The target variable is coded -1 and 1 for dire victory and radiant victory, respectively where "dire" and "radiant" are names of each team. Three features provide game information and the remaining 113 features indicate if a particular hero was played for a given game. The dataset is reasonably sparse as only 10 of 113 possible heroes are chosen in a given game. Furthermore, the data was pre-split into training and test sets and zipped into a single file.

Fundamentally, this is a classification problem where one team wins and one team loses. No ties are allowed. The goal here is not to showcase classification algorithms. Rather, the goal is to introduce HDF5, show how to read/write files to the h5 format with compression, and how to work with HDF5 in pandas.

Caveats: while partial I/O is extremely important from a machine learning perspective, this post will not delve into details. That will come in a future post but you should at least be aware that that capability already exists in HDF5. Additionally, I am including only snippets of my notebook, so for all the gory details look [here](insert link). 

# Setup
```
import pandas as pd
import numpy as np
import h5py
```

# Get Data
The zip file from UCI includes two files, one train and one test. Pandas has zip/unzip functionality but cannot handle zip files with greater than one data object, like in this case. Instead I had to preprocess the data to prep it for pandas. Here is my code:
```
# get zip data from UCI
import requests, zipfile, StringIO
r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip")
z = zipfile.ZipFile(StringIO.StringIO(r.content))
```
Now we can read the datasets into pandas like so:
```
# get train data
X_train = pd.read_csv(z.open('dota2Train.csv'), header=None)

# get test data
X_test = pd.read_csv(z.open('dota2Test.csv'), header=None)
```

# DF to H5 Conversion
Let's convert these dataframes into H5 using three compressors set to max compression. You should update *filepath* to save your data. 
```
compressors = ['blosc', 'bzip2', 'zlib']
for compressor in compressors:
    X_train.to_hdf('filepath_' + str(compressor) + '.h5', 
                   'table', mode='w', append=True, complevel=9, complib=compressor)
    X_test.to_hdf('filepath_' + str(compressor) + '.h5', 
                   'table', mode='w', append=True, complevel=9, complib=compressor)
```

I will show in the next section just how much we were able to compress the data on-disk, but it is important to note that the data gets decompressed when read back into memory. In fact, the memory usage is ever so slightly higher than when we read in the raw file from UCI. So while we can leverage on-disk compression, this method does not compress the data in-memory. Remember that.

# File Sizes
Here, I will calculate the original file size and the file size of each compressed file using blosc, bzip2, and zlib filters. 
```
# get original file size (in MB)
import os
original_size = []
datasets = ['Train', 'Test']
for dataset in datasets:
    original_size.append( round(os.path.getsize("filepath" + str(dataset) + ".csv")/1e6, 2) )
original_data_size = zip(datasets, original_size)

# get compressed file sizes (in MB)
compressed_size = []
for compressor in compressors:
    for dataset in datasets:
        compressed_size.append( round(os.path.getsize("filepath" + dataset + "_" + str(compressor) + ".h5")/1e6, 2) )
compressed_data_size = zip(sorted(['blosc', 'bzip2', 'zlib']*2), ['train', 'test']*3, compressed_size)
```
Barplot showing each for comparison (train set):
```
sns.barplot(['Original', 'blosc', 'bzip2', 'zlib'], train_compressed_size)
plt.ylabel('MB')
plt.title('HDF5 On-Disk Compression: Training Set');
```
**INSERT IMAGE**

Barplot showing each for comparison (test set):
```
sns.barplot(['Original', 'blosc', 'bzip2', 'zlib'], test_compressed_size)
plt.ylabel('MB')
plt.title('HDF5 On-Disk Compression: Test Set');
```
**INSERT IMAGE**

You will notice that bzip2 and zlib compress the data to roughly the same extent. Blosc, on the other hand, does result in significant compression but not to the same extent as bzip2 or zlib. Why is that? Turns out there exists this tradeoff between total compression and read/write times. Zlib and bzip2 are great if your main concern is on-disk storage. If your primary concern is read/write times but you still want to leverage on-disk compression, use blosc. 

# spy()
That's right, I'm bringing Matplotlib's spy() back. 

I use spy() to get a sense of the data's sparsity. Since the data is long and skinny, I transpose it so we can get a better view. Also, I am only capturing the first 1000 rows. Why? Because the dimensions of the dataset are highly skewed which causes the image to get compressed. In other words, the visualization gets crunched so bad that we cannot discern anything useful otherwise.
```
fig = plt.figure(figsize=(15,8))
plt.spy(X_train.transpose().ix[:, :1000]);
```
You can see the data is very sparse in all but the first three features.

# Summary
What are the big takeaways here?

First, if you walk away with nothing else, be aware that HDF5 is a powerful tool that provides on-the-fly compression and partial I/O capabilities. It is so much more than that but knowing that much is a great start.

Secondly, for those newer to zip files, perhaps you learned how to read multiple files zipped together straight into pandas without having to download and unzip anything.

Thirdly, for those new to HDF5, hopefully you learned how to convert dataframes in-memory or files already on your computer to .h5 using compression.

And maybe, just maybe, you can start to see how one could take a sparse matrix that just won't fit into memory and crunch it down so that it can. More on that to come.

# Additional Resources
