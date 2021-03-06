# Conclusion 
---
In this report, we investigate numerous recommendation algorithms in an effort to supply worthwhile recommendations to the user base of Last.fm. Through our initial data exploration of the Last.fm data, we noticed inconstancy's within. We flatten the original relational data model present, to enable the quick development of our recommendation systems. We examine three main recommendation algorithms , a stochastic gradient descent matrix factorization algorithm,  a  regularized stochastic gradient descent matrix factorization algorithm, and a softmax model.  In addition, we employ two different similarity measures, cosine similarity and the dot product to provide more nuance for the discussion of results. We evaluate the aforementioned algorithms at the point of development and through model comparison at the end, utilizing metrics typically used to evaluate information retrieval systems. Finally, we conclude this report by employing popular graph algorithms to understand the listening patterns of the Last.fm users. 

Our most significant results are the following: 

1. Exploratory data analysis revealed that Alternative, ELectronic, Rock and Pop were the most popular genres on the Last.fm website, both in terms of plays and amount of users. Although, the number of user’s a genre has garnered does not necessarily correlate with listening count.  For instance, the "Electronic" genre had highest average play count but the lowest amount of individual listeners  in 2009.  In addition, user activity on the site showed a general increase from 2005 to 2011, the year it was collected. 

2. Our best performing model, SGD_Matrix_Fact_regularized_cosine show-cased the attributes of a good recommendation system i.e. relatively high precision for lower values of k. Although the spread of the evaluation metrics was higher for smaller values of K. 

3. The addition of the regularization  on the SGD Weighted Matrix factorization model proved to be significant. We observed increased performance in recall, precision and hit-rate, which was particularly evident when we drilled down into users according to some defined behaviour (e.g. users with diverse tastes). We believe the vanilla model not only "learned" relevant patterns but also random noise within the test data. Therefore, the regularization of the model prevented this overfilling, which led to a better generalization ability. 

4. The choice of similarity measure proved to be the most significant indicator of performance. As mentioned previously, cosine similarity just takes into account the angle where as dot product takes into account the angle **and** magnitude. The choice of similarity measure is highly domain-specific for in-silico applications. For instance, in information retrieval, the dot product will take the document length into account, whereas the cosine similarity will not. For our use case, the dot product is biased towards mainstream artists i.e. it is more likely to recommend artist with a larger following. We suggest this hypothesis as mainstream artists embeddings tend to have larger norms.  This observation is particularity pronounced when we examine users with varying predispositions to mainstream listenership. We notice that performance of models that incorporated the cosine similarity degraded as user's listening habitats became more mainstream where as the opposite is true for models that incorporated the dot product. Overall, models that used the cosine similarity performed better than those who did not. 

5. There was no distinguishable trends when examining the performance of our models with users of varying activity levels. This is most certainly due to the way we encode user interactions through a binary matrix.

6. The social network of Last.fm does the observe the qualities of a small-world network. We did not observe a small-world phenomenon of users being linked together by a short chain of acquaintances. This is likely due to the fact that Last.fm is not a social network or messaging service. 

7. Label propagation was effective in removing redundancy of user-submitted tags. The newly identified communities of tags  were intuitive.  For instance, 'underground hip hp', 'luso hip hop', 'hip hop under', 'raptuga', and 'hip hop angolano' being clustered together. 


## Future work
There are several possibilities to extend this work in future. Currently, we encode user interactions via a binary matrix.
As we notice no major trends when cross-referencing our models’ performance with user’s activity level, we plan to normalize the listening count feature and evaluate accordingly. The WALS algorithm can take advantage of side information
when recommending items to users. We would be interested in supplying side features such as our newly created super
tags, and the local/betweenness clustering coefficients of users of Graph_1 and Graph_2. After, we plan to examine this
model with our pre-existing models. Presently, we just use precision, recall and hit-rate to evaluate our models. However, these metrics do not completely capture the requirements of a recommender system as outlined in our introductory
paraph. Other metrics we plan to experiment with include coverage, personalization ( 1 - similarity between user’s lists of
recommendations) and intra-list similarity. From the evaluation section, we noticed some models struggled while others
succeeded for different types of users. By segmenting the last.dm data into separate but homogeneous user datasets and
training different recommender systems on those separate data chunks, we hope to potentially improving our results.
Finally, we would be interested in building a hybrid recommendation system. The major weakness of our current collaborative filtering models is that by forgoing the actual characteristics of artists and their songs, the interpretability of
recommendations diminishes. Our idea is to integrate both the features of an artist’s songs (content-based approach) and
user similarity (collaborative filtering)for better music recommendation. 
The [music genome project](https://en.wikipedia.org/wiki/Music_Genome_Project) would serve as the basis for song feature extraction. 
