#Trial example for Collaborative filtering

# Reference - https://ashokharnal.wordpress.com/2014/12/18/worked-out-example-item-based-collaborative-filtering-for-recommenmder-engine/

import pandas as pd
import numpy as np

data = [[241,'u1','m1',2],\
        [222,'u1','m3',3],\
        [276,'u2','m1',5],\
        [273,'u2','m2',2],\
        [200,'u3','m1',3],\
        [229,'u3','m2',3],\
        [231,'u3','m3',1],\
        [239,'u4','m2',2],\
        [286,'u4','m3',2]]

ratings_df = pd.DataFrame(data, columns = ['Id','UserId','MovieId','Rating'])
print('\nStep 1 - DATA: This is our Ratings Table for 4 users and 3 movies')
print(ratings_df)

ratingspvt_df = ratings_df.pivot_table(index='UserId',columns='MovieId',values='Rating')
print('\nStep 2 - CROSSTAB: This is User-Movie Matrix for 4 users and 3 movies')
print(ratingspvt_df)

#MovieId   m1   m2   m3
#UserId                
#u1       2.0  NaN  3.0
#u2       5.0  2.0  NaN
#u3       3.0  3.0  1.0
#u4       NaN  2.0  2.0

ratingspvt_df = ratingspvt_df.fillna(0)
print('\nSTEP 3 - NaN in the User-Movie Matrix is replaced with 0')
print(ratingspvt_df)

#MovieId   m1   m2   m3
#UserId                
#u1       2.0  0.0  3.0
#u2       5.0  2.0  0.0
#u3       3.0  3.0  1.0
#u4       0.0  2.0  2.0

#---------- Lets do Dot Products to get similarity of the users and movies with each other

print('\nSTEP 4 - USER-USER Similarity Matrix')
print('\nSTEP 4a - Multiply ratingspvt_df with transpose of itself')
uu_dotproduct = ratingspvt_df.dot(ratingspvt_df.T)
print(uu_dotproduct)
#UserId    u1    u2    u3   u4
#UserId                       
#u1      13.0  10.0   9.0  6.0
#u2      10.0  29.0  21.0  4.0
#u3       9.0  21.0  19.0  8.0
#u4       6.0   4.0   8.0  8.0

print('\nSTEP 4b - uu_dotproduct/(sqrt of sum of diagnols)')
magnitude_uudp = np.array([np.sqrt(np.diag(uu_dotproduct))])  
uu_sim = uu_dotproduct / magnitude_uudp / magnitude_uudp.T
print(uu_sim)
#UserId        u1        u2        u3        u4
#UserId                                        
#u1      1.000000  0.515026  0.572656  0.588348
#u2      0.515026  1.000000  0.894630  0.262613
#u3      0.572656  0.894630  1.000000  0.648886
#u4      0.588348  0.262613  0.648886  1.000000

#----------------Item-Item Similarity

print('\nSTEP 5 - ITEM-ITEM Similarity Matrix')
print('\nSTEP 5a - Multiply ratingspvt_df with transpose of itself')
ii_dotproduct = ratingspvt_df.T.dot(ratingspvt_df)
print(ii_dotproduct)
#MovieId    m1    m2    m3
#MovieId                  
#m1       38.0  19.0   9.0
#m2       19.0  17.0   7.0
#m3        9.0   7.0  14.0

print('\nSTEP 5b - ii_dotproduct/(sqrt of sum of diagnols)')
magnitude_iidp = np.array([np.sqrt(np.diag(ii_dotproduct))])  
ii_sim = ii_dotproduct / magnitude_iidp / magnitude_iidp.T
print(ii_sim)
#MovieId        m1        m2        m3
#MovieId                              
#m1       1.000000  0.747545  0.390199
#m2       0.747545  1.000000  0.453743
#m3       0.390199  0.453743  1.000000

