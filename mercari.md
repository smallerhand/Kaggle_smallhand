## Workflow stages

1. Question or problem definition.
2. Acquire data.
3. Analyze by describing data.
4. Analyze, identify patterns, and explore the data.
5. Model, predict and solve the problem.
6. Visualize, report, and present the problem solving steps and final solution.
7. Supply or submit the results.

The workflow indicates general sequence of how each stage may follow the other. However there are use cases with exceptions.

- We may combine mulitple workflow stages. We may analyze by visualizing data.
- Perform a stage earlier than indicated. We may analyze data before and after wrangling.
- Perform a stage multiple times in our workflow. Visualize stage may be used multiple times.
- Drop a stage altogether. We may not need supply stage to productize or service enable our dataset for a competition.

## Question or problem definition.

- to build an algorithm that automatically suggests the right product prices. 
- be provided user-inputted text descriptions of their products, including details like product category name, brand name, and item condition.

https://www.kaggle.com/c/mercari-price-suggestion-challenge/overview/description


```python
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os

# text mining
import nltk
import nltk.corpus
nltk.download('punkt')

# importing word_tokenize from nltk
from nltk.tokenize import word_tokenize

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\skim\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

## Acquire data

The Python Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. We also combine these datasets to run certain operations on both datasets together.


```python
train = pd.read_csv('C:/Users/skim/Desktop/kaggle/mercari/data/mercari-price-suggestion-challenge/train.tsv', sep='\t')
test = pd.read_csv('C:/Users/skim/Desktop/kaggle/mercari/data/mercari-price-suggestion-challenge/test.tsv', sep='\t')
combine = [train, test]
```

## Analyze by describing data


```python
print(train.shape)
print(test.shape)
print(train.columns.values)
```

    (1482535, 8)
    (693359, 7)
    ['train_id' 'name' 'item_condition_id' 'category_name' 'brand_name'
     'price' 'shipping' 'item_description']
    


```python
print(train.head())
```

       train_id                                 name  item_condition_id  \
    0         0  MLB Cincinnati Reds T Shirt Size XL                  3   
    1         1     Razer BlackWidow Chroma Keyboard                  3   
    2         2                       AVA-VIV Blouse                  1   
    3         3                Leather Horse Statues                  1   
    4         4                 24K GOLD plated rose                  1   
    
                                           category_name brand_name  price  \
    0                                  Men/Tops/T-shirts        NaN   10.0   
    1  Electronics/Computers & Tablets/Components & P...      Razer   52.0   
    2                        Women/Tops & Blouses/Blouse     Target   10.0   
    3                 Home/Home Décor/Home Décor Accents        NaN   35.0   
    4                            Women/Jewelry/Necklaces        NaN   44.0   
    
       shipping                                   item_description  
    0         1                                 No description yet  
    1         0  This keyboard is in great condition and works ...  
    2         1  Adorable top with a hint of lace and a key hol...  
    3         1  New with tags. Leather horses. Retail for [rm]...  
    4         0          Complete with certificate of authenticity  
    


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1482535 entries, 0 to 1482534
    Data columns (total 8 columns):
     #   Column             Non-Null Count    Dtype  
    ---  ------             --------------    -----  
     0   train_id           1482535 non-null  int64  
     1   name               1482535 non-null  object 
     2   item_condition_id  1482535 non-null  int64  
     3   category_name      1476208 non-null  object 
     4   brand_name         849853 non-null   object 
     5   price              1482535 non-null  float64
     6   shipping           1482535 non-null  int64  
     7   item_description   1482531 non-null  object 
    dtypes: float64(1), int64(3), object(4)
    memory usage: 90.5+ MB
    

## Analyze by pivoting features


```python
train[['name', 'price']].groupby(['name'], as_index=False).mean().sort_values(by='price', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>706440</th>
      <td>NEW Chanel WOC Caviar Gold Hardware</td>
      <td>2009.0</td>
    </tr>
    <tr>
      <th>714071</th>
      <td>NEW-Chanel Boy Wallet o Chain WOC Caviar</td>
      <td>2006.0</td>
    </tr>
    <tr>
      <th>316772</th>
      <td>David Yurman Wheaton ring</td>
      <td>2004.0</td>
    </tr>
    <tr>
      <th>270634</th>
      <td>Chanel Classic Jumbo Single flap bag</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <th>270631</th>
      <td>Chanel Chevron Fuschia Pink 2</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>944563</th>
      <td>Sea Turtle Charm Necklace</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>406710</th>
      <td>Frozen Elsa Dress</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>779421</th>
      <td>Nike Dri-Fit High Power Speed Tights</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>698318</th>
      <td>Mossimo denium shorts</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>489185</th>
      <td>Infant Carter's "My First Thanksgiving"</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1225273 rows × 2 columns</p>
</div>




```python
train[['item_condition_id', 'price']].groupby(['item_condition_id'], as_index=False).mean().sort_values(by='price', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_condition_id</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>31.703859</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>27.563225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>26.540711</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>26.486967</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>24.349212</td>
    </tr>
  </tbody>
</table>
</div>




```python
train[['category_name', 'price']].groupby(['category_name'], as_index=False).mean().sort_values(by='price', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category_name</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>963</th>
      <td>Vintage &amp; Collectibles/Antique/Furniture</td>
      <td>195.000000</td>
    </tr>
    <tr>
      <th>153</th>
      <td>Handmade/Bags and Purses/Clutch</td>
      <td>180.222222</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Electronics/Computers &amp; Tablets/Laptops &amp; Netb...</td>
      <td>177.089176</td>
    </tr>
    <tr>
      <th>681</th>
      <td>Kids/Strollers/Standard</td>
      <td>163.666667</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Electronics/Computers &amp; Tablets/Desktops &amp; All...</td>
      <td>149.329412</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Handmade/Knitting/Doll</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>215</th>
      <td>Handmade/Dolls and Miniatures/Artist Bears</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>590</th>
      <td>Kids/Diapering/Washcloths &amp; Towels</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>280</th>
      <td>Handmade/Jewelry/Clothing</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>189</th>
      <td>Handmade/Children/Jewelry</td>
      <td>3.600000</td>
    </tr>
  </tbody>
</table>
<p>1287 rows × 2 columns</p>
</div>




```python
train[['brand_name', 'price']].groupby(['brand_name'], as_index=False).mean().sort_values(by='price', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brand_name</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1222</th>
      <td>Demdaco</td>
      <td>429.000000</td>
    </tr>
    <tr>
      <th>3465</th>
      <td>Proenza Schouler</td>
      <td>413.250000</td>
    </tr>
    <tr>
      <th>346</th>
      <td>Auto Meter</td>
      <td>344.000000</td>
    </tr>
    <tr>
      <th>3187</th>
      <td>Oris</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>2656</th>
      <td>MCM Worldwide</td>
      <td>289.173913</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3819</th>
      <td>Scunci</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>3381</th>
      <td>Play MG</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>2285</th>
      <td>Kae Argatherapie</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>Gossip Girl</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>777</th>
      <td>CM Style Fashion</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>4809 rows × 2 columns</p>
</div>




```python
train[['shipping', 'price']].groupby(['shipping'], as_index=False).mean().sort_values(by='price', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shipping</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>30.111778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22.567726</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.hist(train.price)
```




    (array([1.472289e+06, 8.335000e+03, 1.169000e+03, 3.990000e+02,
            1.780000e+02, 6.500000e+01, 4.700000e+01, 2.100000e+01,
            1.600000e+01, 1.600000e+01]),
     array([   0. ,  200.9,  401.8,  602.7,  803.6, 1004.5, 1205.4, 1406.3,
            1607.2, 1808.1, 2009. ]),
     <a list of 10 Patch objects>)




![png](mercari_files/mercari_16_1.png)


## Add some columns and see deeply


```python
train['log_price'] = np.log1p(train['price'])
plt.hist(train.log_price)
```




    (array([8.74000e+02, 1.87030e+04, 1.93836e+05, 6.20246e+05, 4.46867e+05,
            1.53715e+05, 3.83370e+04, 8.41300e+03, 1.34000e+03, 2.04000e+02]),
     array([0.      , 0.760589, 1.521178, 2.281767, 3.042356, 3.802945,
            4.563534, 5.324123, 6.084712, 6.845301, 7.60589 ]),
     <a list of 10 Patch objects>)




![png](mercari_files/mercari_18_1.png)



```python
train['category1'] = train.category_name.str.split('/').str[0]
train['category2'] = train.category_name.str.split('/').str[1]
train['category3'] = train.category_name.str.split('/').str[2]
```


```python
print(len(pd.unique(train['category1'])))
sns.countplot(x="category1", data=train)
plt.xticks(rotation=90)
plt.show()
```

    11
    


![png](mercari_files/mercari_20_1.png)



```python
train[['category1', 'price']].groupby(['category1'], as_index=False).mean().sort_values(by='price', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category1</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Electronics</td>
      <td>35.173922</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Men</td>
      <td>34.708614</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Women</td>
      <td>28.885496</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vintage &amp; Collectibles</td>
      <td>27.339426</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sports &amp; Outdoors</td>
      <td>25.532219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Home</td>
      <td>24.536599</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Other</td>
      <td>20.809817</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kids</td>
      <td>20.642315</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Beauty</td>
      <td>19.671536</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Handmade</td>
      <td>18.156572</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_Women = train[train['category1']=='Women']
train_Women[['category2', 'price']].groupby(['category2'], as_index=False).mean().sort_values(by='price', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category2</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>Women's Handbags</td>
      <td>58.201648</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Shoes</td>
      <td>35.975610</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Coats &amp; Jackets</td>
      <td>34.041360</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Women's Accessories</td>
      <td>30.930531</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dresses</td>
      <td>29.445015</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Athletic Apparel</td>
      <td>28.844614</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jewelry</td>
      <td>28.058633</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sweaters</td>
      <td>26.503293</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Other</td>
      <td>26.012665</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jeans</td>
      <td>25.885614</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Swimwear</td>
      <td>21.838491</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Skirts</td>
      <td>21.546541</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Maternity</td>
      <td>21.111971</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Pants</td>
      <td>19.651047</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Suits &amp; Blazers</td>
      <td>19.193616</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tops &amp; Blouses</td>
      <td>18.237514</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Underwear</td>
      <td>18.097813</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_Women_Athletic = train_Women[train_Women['category2']=='Athletic Apparel']
train_Women_Athletic[['category3', 'price']].groupby(['category3'], as_index=False).mean().sort_values(by='price', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category3</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>Tracksuits &amp; Sweats</td>
      <td>36.702572</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Jackets</td>
      <td>34.797082</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pants, Tights, Leggings</td>
      <td>34.392733</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Snowsuits &amp; Bibs</td>
      <td>33.507937</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Vests</td>
      <td>32.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Other</td>
      <td>31.631902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jerseys</td>
      <td>23.670157</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Skirts, Skorts &amp; Dresses</td>
      <td>22.560956</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shirts &amp; Tops</td>
      <td>22.450672</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sports Bras</td>
      <td>19.718043</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Shorts</td>
      <td>18.547650</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Socks</td>
      <td>14.176233</td>
    </tr>
  </tbody>
</table>
</div>




```python
cnt = 0
for i in train_Women_Athletic.item_description:
    print(i[:100])
    print("_"*100)
    cnt += 1
    if cnt > 100:
        break
```

    NWT Victoria's Secret ULTIMATE SPORT BRA -MAXIMUM SUPPORT SIZE 34ddd
    ____________________________________________________________________________________________________
    Worn one time. Excellent condition
    ____________________________________________________________________________________________________
    No description yet
    ____________________________________________________________________________________________________
    One size fits sizes 2-12 92% polyester 8% spandex Super soft!! Capri leggings High waist 1in elastic
    ____________________________________________________________________________________________________
    Brand new with tag and bag. French bulldog leggings. Hard to find this color background. Super soft!
    ____________________________________________________________________________________________________
    Distressed Levi high waist jeans. Size XS, 12 slim in kids size
    ____________________________________________________________________________________________________
    Highwaist distressed denim shorts size 3 The shorts I wear in most of my images
    ____________________________________________________________________________________________________
    Under Armour half zip jacket in awesome like new condition! Size med.
    ____________________________________________________________________________________________________
    Watercolor Inspire crop. No pilling or stickiness whatsoever. Some of the pink has bled into the whi
    ____________________________________________________________________________________________________
    Cute boutique brand suit. Floral Print . Polyester & spandex material UNDER BRAND FOR VIEWING
    ____________________________________________________________________________________________________
    I love these, but I got them as a mystery pair and they're not my size. :-( I am always happy to com
    ____________________________________________________________________________________________________
    Size 10, worn but good condition.
    ____________________________________________________________________________________________________
    Brand new. Still in package. Black background with orange tigger.
    ____________________________________________________________________________________________________
    New fall print. Deep purple background with Siamese cats, hints of mustard, aqua and white. Tall and
    ____________________________________________________________________________________________________
    2 pair black Capri tights.
    ____________________________________________________________________________________________________
    OS sugar skulls leggings Red background with little flowers and skulls Brand new
    ____________________________________________________________________________________________________
    Brand new, highly sought after unicorn. Black background with white birds on a wire. These are so pr
    ____________________________________________________________________________________________________
    NWT LuLaRoe OS (One Size) leggings. Purple background with storks carrying light blue and light pink
    ____________________________________________________________________________________________________
    NEW!! Never worn or tried on. Lularoe buttery soft leggings HTF OS LuLaRoe Rudolph the Lularoe unico
    ____________________________________________________________________________________________________
    No description yet
    ____________________________________________________________________________________________________
    Brand new!!
    ____________________________________________________________________________________________________
    Good condition has small pocket on back and zippers on bottom. Great workout leggings
    ____________________________________________________________________________________________________
    Used, 2 small holes shown in pic 4 on back of thigh bellow butt, minor pilling between legs, size M,
    ____________________________________________________________________________________________________
    Dri-Fit The bottom of the N wore off a little
    ____________________________________________________________________________________________________
    One black top, one plum top- both size large
    ____________________________________________________________________________________________________
    LuLaRoe Leggings One Size Halloween Frankenstein Print New Without Tag Never Worn Or Washed
    ____________________________________________________________________________________________________
    No description yet
    ____________________________________________________________________________________________________
    • FREE SHIPPING! • brand new with tags • size medium
    ____________________________________________________________________________________________________
    3 new pairs pink socks and free ty gift and free shipping
    ____________________________________________________________________________________________________
    Lululemon Run Inspire Crop Skinny Women's Pants 6 FAST SHIP! Great. Measurements taken by hand: Wais
    ____________________________________________________________________________________________________
    -Camouflage colors -Size small -Brand new Freddy Jeggings -100% authentic **If you have any question
    ____________________________________________________________________________________________________
    Smoke and pet free home. Fast shipping.
    ____________________________________________________________________________________________________
    BRAND NWOT NFL Team Apparel Women's NY Giants Fitted Lace Up Jersey Top Red Lace-Up and Bling Letter
    ____________________________________________________________________________________________________
    Neon green UA fitted tank NWT! Womens M CHECK OUT MY CLOSET for other items like nike shoes, lululem
    ____________________________________________________________________________________________________
    Like new
    ____________________________________________________________________________________________________
    Medium vs pink boyfriend pants, well loved
    ____________________________________________________________________________________________________
    Black sweatpants by Victoria's Secret "Angel" loungewear. Drawstring waist with side pockets. Only w
    ____________________________________________________________________________________________________
    New with tag Tall and Curvy legging from Lularoe. Made in china
    ____________________________________________________________________________________________________
    Size xs very goodcondition boot cut sweatpants. Has pockets on back side.
    ____________________________________________________________________________________________________
    Black with unique colors around the sides.
    ____________________________________________________________________________________________________
    Brand new floral camera leggings
    ____________________________________________________________________________________________________
    Women's Cobalt blue down jacket. Gently Used. Very warm and snuggly on those cold mornings! Generous
    ____________________________________________________________________________________________________
    Green like new Nike shorts! Great condition! Excellent for a jog or a bum day! Offers are accepted! 
    ____________________________________________________________________________________________________
    One size fits small-large. Soft brush knit leggings are amazingly comfortable! 92 % polyester, 8 % s
    ____________________________________________________________________________________________________
    Brand New! Only removed from packaging for photos. Made in Vietnam.
    ____________________________________________________________________________________________________
    Medium in women's (⚡️⚡️I have a lot of Nike deals, visit my closet ⚡️⚡️)
    ____________________________________________________________________________________________________
    LuLaRoe OS Leggings made in China. VGUC I bought used but never wore. Close up pictures included for
    ____________________________________________________________________________________________________
    Brand new with tag. Made in Vietnam
    ____________________________________________________________________________________________________
    Wildfox Couture SweetHeart Cutie Shorts Womens Sz S Rare BBJ Sleep
    ____________________________________________________________________________________________________
    Purple gymshark tank top. Small. Racerback. Somewhat see through. Slits on side. Never worn
    ____________________________________________________________________________________________________
    Like new, size S FIRM PLS
    ____________________________________________________________________________________________________
    Super cute sports bra bundle- cobalt blue and black- cross back detail. Too small for me. Never worn
    ____________________________________________________________________________________________________
    Tan with pretty flowers
    ____________________________________________________________________________________________________
    No description yet
    ____________________________________________________________________________________________________
    Bought way too many and would like to get rid of my least favorite ones. Brand new. Just took out of
    ____________________________________________________________________________________________________
    Wear on the butt (shown in picture) black with forest flower design. Tween. Retail for 25
    ____________________________________________________________________________________________________
    Aeropostale brand | Size: XL | Cotton blend | Legging style | Full length | Fold over band is blue a
    ____________________________________________________________________________________________________
    Gently used leather jacket. It's similar to the one Hayley Williams has (idk if it's the same exact 
    ____________________________________________________________________________________________________
    New- size 4 Super soft!!
    ____________________________________________________________________________________________________
    Worn a few times, great condition. Unicorn print.
    ____________________________________________________________________________________________________
    navy blue in color with elastic waist band and leg holes
    ____________________________________________________________________________________________________
    Victoria's secret pink quarter zip jacket in awesome like new condition! Size small.
    ____________________________________________________________________________________________________
    Brand new with tags. Black background with tan/orange polka dots. Very cute!
    ____________________________________________________________________________________________________
    New
    ____________________________________________________________________________________________________
    Lot of LulaRoe leggings and a small perfect T All OS None worn just tried on
    ____________________________________________________________________________________________________
    Victoria secret pink shirt bling size small front is open on the neck line
    ____________________________________________________________________________________________________
    Brand New! Comfortable and flattering full on Luon fabric. Soft high stretch with added support and 
    ____________________________________________________________________________________________________
    Lularoe tc NWOT floral leggings very pretty
    ____________________________________________________________________________________________________
    No description yet
    ____________________________________________________________________________________________________
    Perfect blue jean shorts! Go right to your belly button, cover all of your cheeks! Perfect condition
    ____________________________________________________________________________________________________
    This is brand new never worn before. The back is completely mesh and looks very cute with a sports b
    ____________________________________________________________________________________________________
    Size 2 excellent condition Grey color
    ____________________________________________________________________________________________________
    Lululemon black wunder under crops in used condition! They have pilling. Size 4!
    ____________________________________________________________________________________________________
    No description yet
    ____________________________________________________________________________________________________
    Hollister Sweatpants size M 6/10 condition. Normal wear. Please make sure you know your size. Size M
    ____________________________________________________________________________________________________
    Like New Lululemon Herringbone Think Fast pullover Sz 12. Tag is removed but def a size 12. Quarter 
    ____________________________________________________________________________________________________
    Extra large joggers from under armour. Too big for me and I got them as a gift. The tag has been cut
    ____________________________________________________________________________________________________
    black zip up jacket by adidas. has pockets and a hood! hoodie strings intact. no flaws - i just have
    ____________________________________________________________________________________________________
    BUTTERY SOFT BLACK WHITE GRAY AZTEC LEGGINGS   Size: Plus Size (Similar to Tall & Curvy) Numeric Siz
    ____________________________________________________________________________________________________
    These are brand new. I got another color way and love those more so I'm going to sell these to you!
    ____________________________________________________________________________________________________
    New with tag one size leggings from Lularoe. You will receive the legging on the picture.
    ____________________________________________________________________________________________________
    Under Armour Women's Tank Top Brand new size small
    ____________________________________________________________________________________________________
    Pink shirt, size L 100% cotton
    ____________________________________________________________________________________________________
    Smoke free, pet free home.
    ____________________________________________________________________________________________________
    Coral color, never worn, short sleeve
    ____________________________________________________________________________________________________
    NWT Women's Nike dri-fit running shorts. Size large. Drawstring feature for elastic waistband. Light
    ____________________________________________________________________________________________________
    NWOT LuLaRoe leggings solid teal. Tall and curvy fits sizes 12-22. Way below retail.
    ____________________________________________________________________________________________________
    New Without Tags Free Shipping 9 for [rm]
    ____________________________________________________________________________________________________
    LuLaRoe one size vintage paisley unicorn Hard to find print No excessive signs of wear
    ____________________________________________________________________________________________________
    No description yet
    ____________________________________________________________________________________________________
    BNWT, never worn or tried on Little umbrellas and small unicorns on them TC sizes 12-22 FREE SHIP *N
    ____________________________________________________________________________________________________
    Worn once. Minimal fading. Size 27
    ____________________________________________________________________________________________________
    There super cute but there a little see through if you want them for the gym especially if you squat
    ____________________________________________________________________________________________________
    Like new condition, quarter zip, women's Medium, super cute *Bundle on any item and get same amount 
    ____________________________________________________________________________________________________
    Size small Only worn once
    ____________________________________________________________________________________________________
    Worn and washed once per directions. Black tc lularoe leggings made in Indonesia. Fits women's sizes
    ____________________________________________________________________________________________________
    Size 4 Excellent flawless condition RELAXED SENSATION Gives you the ultimate feeling of nothing in y
    ____________________________________________________________________________________________________
    New lularoe tc leggings black and white
    ____________________________________________________________________________________________________
    Size XS women's NEW with tags Retail [rm] firm price no Free shipping I have a lot of Nike deals, vi
    ____________________________________________________________________________________________________
    OS mermaid leggings Excellent used condition Htf print!
    ____________________________________________________________________________________________________
    Brand new LulaRoe One size grey fox face leggings ~smoke free ~quick shipping ~please ask any questi
    ____________________________________________________________________________________________________
    

## From Descriptions


```python
print(train_Women_Athletic.item_description.values)

```

    ["NWT Victoria's Secret ULTIMATE SPORT BRA -MAXIMUM SUPPORT SIZE 34ddd"
     'Worn one time. Excellent condition' 'No description yet' ...
     'NWOT - Blue - size 8'
     '▪️NWOT ▪️Perfect Condition ▪️Barely been worn ▪️No flaws ▪️Not Nike from Stadium Athletics ▪️No pockets, great for leisure wear'
     "Purple and Paisley Victoria's Secret Tankini Size Large. Worn a handful of times. Excellent Condition Free Shipping!"]
    


```python
# Passing the string text into word tokenize for breaking the sentences
token = word_tokenize(train_Women_Athletic.item_description.values[1])

# finding the frequency distinct in the tokens
# Importing FreqDist library from nltk and passing token into FreqDist
from nltk.probability import FreqDist
fdist = FreqDist(token)
fdist

# To find the frequency of top 10 words
fdist1 = fdist.most_common(10)
fdist1
```




    [('Worn', 1),
     ('one', 1),
     ('time', 1),
     ('.', 1),
     ('Excellent', 1),
     ('condition', 1)]



## Model, predict and solve

Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:

- Logistic Regression
- KNN or k-Nearest Neighbors
- Support Vector Machines
- Random Forrest
- neural network


```python
X_train = train.drop('price', axis=1)
Y_train = train['price']
X_test  = test.drop('test_id', axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
```




    ((1482535, 11), (1482535,), (693359, 6))




```python
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-52-91ca742411e9> in <module>
          2 
          3 svc = SVC()
    ----> 4 svc.fit(X_train, Y_train)
          5 Y_pred = svc.predict(X_test)
          6 acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\sklearn\svm\_base.py in fit(self, X, y, sample_weight)
        146         X, y = check_X_y(X, y, dtype=np.float64,
        147                          order='C', accept_sparse='csr',
    --> 148                          accept_large_sparse=False)
        149         y = self._validate_targets(y)
        150 
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\sklearn\utils\validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)
        753                     ensure_min_features=ensure_min_features,
        754                     warn_on_dtype=warn_on_dtype,
    --> 755                     estimator=estimator)
        756     if multi_output:
        757         y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        529                     array = array.astype(dtype, casting="unsafe", copy=False)
        530                 else:
    --> 531                     array = np.asarray(array, order=order, dtype=dtype)
        532             except ComplexWarning:
        533                 raise ValueError("Complex data not supported\n"
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\numpy\core\_asarray.py in asarray(a, dtype, order)
         83 
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86 
         87 
    

    ValueError: could not convert string to float: 'Wallets'



```python
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
```


```python
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-25-2629706921d5> in <module>
    ----> 1 g = sns.FacetGrid(train, col='price')
          2 g.map(plt.hist, 'shipping', bins=20)
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\seaborn\axisgrid.py in __init__(self, data, row, col, hue, col_wrap, sharex, sharey, height, aspect, palette, row_order, col_order, hue_order, hue_kws, dropna, legend_out, despine, margin_titles, xlim, ylim, subplot_kws, gridspec_kws, size)
        386 
        387         # Make the axes look good
    --> 388         fig.tight_layout()
        389         if despine:
        390             self.despine()
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\cbook\deprecation.py in wrapper(*args, **kwargs)
        356                 f"%(removal)s.  If any parameter follows {name!r}, they "
        357                 f"should be pass as keyword, not positionally.")
    --> 358         return func(*args, **kwargs)
        359 
        360     return wrapper
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\figure.py in tight_layout(self, renderer, pad, h_pad, w_pad, rect)
       2487 
       2488         if renderer is None:
    -> 2489             renderer = get_renderer(self)
       2490 
       2491         kwargs = get_tight_layout_figure(
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\tight_layout.py in get_renderer(fig)
        220 
        221         if canvas and hasattr(canvas, "get_renderer"):
    --> 222             renderer = canvas.get_renderer()
        223         else:  # Some noninteractive backends have no renderer until draw time.
        224             cbook._warn_external("tight_layout: falling back to Agg renderer")
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\backends\backend_agg.py in get_renderer(self, cleared)
        402                           and getattr(self, "_lastKey", None) == key)
        403         if not reuse_renderer:
    --> 404             self.renderer = RendererAgg(w, h, self.figure.dpi)
        405             self._lastKey = key
        406         elif cleared:
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\backends\backend_agg.py in __init__(self, width, height, dpi)
         90         self.width = width
         91         self.height = height
    ---> 92         self._renderer = _RendererAgg(int(width), int(height), dpi)
         93         self._filter_renderers = []
         94 
    

    ValueError: Image size of 178848x216 pixels is too large. It must be less than 2^16 in each direction.



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\IPython\core\formatters.py in __call__(self, obj)
        339                 pass
        340             else:
    --> 341                 return printer(obj)
        342             # Finally look for special method names
        343             method = get_real_method(obj, self.print_method)
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\IPython\core\pylabtools.py in <lambda>(fig)
        246 
        247     if 'png' in formats:
    --> 248         png_formatter.for_type(Figure, lambda fig: print_figure(fig, 'png', **kwargs))
        249     if 'retina' in formats or 'png2x' in formats:
        250         png_formatter.for_type(Figure, lambda fig: retina_figure(fig, **kwargs))
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\IPython\core\pylabtools.py in print_figure(fig, fmt, bbox_inches, **kwargs)
        130         FigureCanvasBase(fig)
        131 
    --> 132     fig.canvas.print_figure(bytes_io, **kw)
        133     data = bytes_io.getvalue()
        134     if fmt == 'svg':
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\backend_bases.py in print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format, bbox_inches, **kwargs)
       2076                         functools.partial(
       2077                             print_method, dpi=dpi, orientation=orientation),
    -> 2078                         draw_disabled=True)
       2079                     self.figure.draw(renderer)
       2080                     bbox_artists = kwargs.pop("bbox_extra_artists", None)
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\backend_bases.py in _get_renderer(figure, print_method, draw_disabled)
       1537     with cbook._setattr_cm(figure, draw=_draw):
       1538         try:
    -> 1539             print_method(io.BytesIO())
       1540         except Done as exc:
       1541             renderer, = figure._cachedRenderer, = exc.args
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\backends\backend_agg.py in print_png(self, filename_or_obj, metadata, pil_kwargs, *args, **kwargs)
        512         }
        513 
    --> 514         FigureCanvasAgg.draw(self)
        515         if pil_kwargs is not None:
        516             from PIL import Image
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\backends\backend_agg.py in draw(self)
        386         Draw the figure using the renderer.
        387         """
    --> 388         self.renderer = self.get_renderer(cleared=True)
        389         # Acquire a lock on the shared font cache.
        390         with RendererAgg.lock, \
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\backends\backend_agg.py in get_renderer(self, cleared)
        402                           and getattr(self, "_lastKey", None) == key)
        403         if not reuse_renderer:
    --> 404             self.renderer = RendererAgg(w, h, self.figure.dpi)
        405             self._lastKey = key
        406         elif cleared:
    

    c:\users\skim\appdata\local\programs\python\python37\lib\site-packages\matplotlib\backends\backend_agg.py in __init__(self, width, height, dpi)
         90         self.width = width
         91         self.height = height
    ---> 92         self._renderer = _RendererAgg(int(width), int(height), dpi)
         93         self._filter_renderers = []
         94 
    

    ValueError: Image size of 178848x216 pixels is too large. It must be less than 2^16 in each direction.



    <Figure size 178848x216 with 828 Axes>



```python
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
```

## Model evaluation

We can now rank our evaluation of all the models to choose the best one for our problem. While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set.


```python

```


```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
```


```python
submission = pd.DataFrame({
        "test_id": test["test_id"],
        "price": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)
```
