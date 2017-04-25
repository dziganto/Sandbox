**(insert cover image)**

# Introduction
In my last post [Sparse Matrices For Efficient Machine Learning](insert link), I showcased methods and a workflow for converting matrices with lots of zero values into a sparse matrix with Scipy. This did two things:

 1. compressed the in-memory footprint of the data matrix 
 2. sped up many machine learning routines

I ended that post with a challenge: **if the original data matrix won’t fit into memory in the first place, can you think of a way to convert it to a sparse matrix anyway?**

To solve this problem we need to think bigger. We need to step away from in-memory techniques and instead delve into on-disk methods.

In this vein, allow me to introduce Hierarchical Data Format version 5 (HDF5), an extremely powerful tool rife with capabilities. As best summarized in [Python and HDF5](http://shop.oreilly.com/product/0636920030249.do) 
> "HDF5 is just about perfect if you make minimal use of relational features and have a need for very high performance, partial I/O, hierarchical organization, and arbitrary
metadata".

I highly recommend that book and the links in the *Additional Resources" section if you want to learn more but in this post I will specifically focus on two areas that are highly pertinent to machine learning workflows:

1. on-disk data compression 
2. partial I/O

Before digging in, allow me to introduce the dataset we will be working with.

# Dataset: Dota 2

**Insert Dota2 Image**

Dota 2 is a popular computer game published by Valve Corporation. Two teams consisting of five players are forged from among 113 heros, each with unique strengths and weaknesses. Players and teams gain experience and items during the course of the game, which ends when one team destroys the "Ancient", a large structure in the opposing team's base.

I chose the [Dota2 Games Results Data Set](https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results#) for two reasons:

1. it is real-world data

2) many people are interested in gaming

The dataset captures information for all games played in a space of 2 hours on the 13th of August, 2016. Specifically, the dataset consists of a target variable and 116 features. The target variable is coded -1 and 1 for dire victory and radiant victory, respectively where "dire" and "radiant" are names of each team. Three features provide game information and the remaining 113 features indicate if a particular hero was played for a given game. The dataset is reasonably sparse as only 10 of 113 possible heroes are chosen in a given game. Furthermore, the data was pre-split into training and test sets and zipped into a single file.

Fundamentally, this is a classification problem where one team wins and one team loses. No ties are allowed. The goal here is not to showcase classification algorithms. Rather, the goal is to introduce two very powerful attributes of HDF5: 

1. on-disk data compression 
2. partial I/O 

These two properties are immensely useful to anyone using machine learning.

# Data Compression
- how to convert pandas df to hdf5 - explanation of compression options - show
compression examples


# Partial I/O
- problem description - example


# Summary
- data compression - partial I/O


# Additional Resources
 
 
# Extra
Common file formats like CSV or JSON can be converted to H5. Why convert a file
format? HDF5 comes with many compressors and compression options that allow us
to compress the data size. It also allows us to store data in a standard format
accessible to anyone with HDF5. Resolving
[endianess](https://www.cs.umd.edu/class/sum2003/cmsc311/Notes/Data/endian.html)
is just one example of a problem solved by HDF5. There are many others but that
is beyond the scope of this post.
