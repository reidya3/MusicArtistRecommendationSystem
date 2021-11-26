# Conclusion 
---
In this report, we investigate numerous recommendation algorithms in an effort to supply worthwhile recommendations to the user base of Last.fm. Through our initial data exploration of the Last.fm data, we noticed inconstancy's within. We flatten the original relational data model present, to enable the quick development of our recommendation systems. We examine three main recommendation algorithms , a stochastic gradient descent matrix factorization algorithm,  a  regularized stochastic gradient descent matrix factorization algorithm, and a softmax model.  In addition, we employ two different similarity measures, cosine similarity and the dot product to provide more nuance for the discussion of results. We evaluate the aforementioned algorithms at the point of development and through a model comparison at the end, utilising metrics typically used to evaluate information retrieval systems. Finally, we conclude this report by employing popular graph algorithms, to understand the listening patterns of the Last.fm users. 

Our most significant results are the following: 
1. Exploratory data analysis revealed that Alternative, ELectronic, Rock and Pop were the most popular genres on the Last.fm website, both in terms of plays and amount of users. Although, the number of user’s a genre has garnered does not necessarily correlate with listening count.  For instance, the "Electronic" genre had highest average play count but the lowest amount of individual listeners  in 2009.  In addition, user activity on the site showed a general increase from 2005 to 2011, the year it was collected. 

2. Our best performing model, SGD_Matrix_Fact_regularized_cosine show-cased the attributes of a good recommendation system i.e. relatively high precision for lower values of k. Although the spread of the evaluation metrics was higher for smaller values of K. 

3. The addition of the regularization  on the SGD Matrix factorization model proved to be significant. We observed increased performance in recall, precision and hit-rate, which was particularly evident when we drilled down into users according to some defined behaviour (e.g. users with diverse tastes). We believe the vanilla model not only "learned" relevant patterns but also random noise within the test data. Therefore, the regularization of the model prevented this overfilling, which led to a better generalization ability. 

4. The choice of similarity measure proved to be the most significant indicator of performance. As mentioned previously, cosine similarity just takes into account the angle where as dot product takes into account the angle **and** magnitude. The choice of similarity measure is highly domain-specific for in-silico applications. For instance, in information retrieval, the dot product will take the document length into account, whereas the cosine similarity will not. For our use case, the dot product is biased towards mainstream artists i.e. it is more likely to recommend artist with a larger following. We suggest this hypothesis as mainstream artists embeddings tend to have larger norms.  This observation is particularity pronounced when we examine users with varying predispositions to mainstream listenership. We notice that performance of models that incorporated the cosine similarity degraded as user's listening habitats became more mainstream where as the opposite is true for models that incorporated the dot product. 

5. There was no distinguishable trends when examining the performance of our models with users of varying activity levels. This is most certainly due to the way we encode user interactions through a binary matrix.

5 



6. Using label


1. Exploratory data analysis of the various studies contradicts the general assumption of IGT, i.e., the preference of healthy individuals to seek long-term reward. Rather than picking the two advantageous decks (C, D), ad-hoc analysis demonstrated participants generally prefer one of the advantageous (deck D) and one of the disadvantageous (deck B). Possible reasons for this observed discrepancy may be found in the particular payoff scheme of the study and the resulting inter-study biases. Alternatively, we hypothesis that healthy individuals are influenced by **both** long term reward and immediate gain/loss frequency.
2. Similarly, we observed a high inter-study and inter-individual variability in IGT performance in healthy participants. Participant variability could be due to divergent psychological attributes of the healthy participants such as learn behaviour, a propensity to gambling, impulsivity or different decision-making strategies. In addition, different IGT versions may explain the inter-study discrepancy. 
3. Although both heroin and amphetamine users display poor decision making, the average heroin user displays a poorer take-home reward (approx. $200 difference).  Different classes of drugs might have different effects on decision-making behaviour. As mentioned previously,  pre-clinical trails concluded that stimulant and opiate users display different behavioural effects. Stimulants tend to produce arousing and activating effects. In contrast, opiates produce mixed inhibitory and excitatory effects.
3. K-means could reasonably group all unhealthy individuals into a single cluster, but healthy individuals were distributed evenly. K-means performed poorly when we examined the clusters by study. 
4. The devised federated k-means algorithm resulted in a 10% increase in the sum square error.

## Future work

- activity level - change from binary to normalised count. 
- tags wals algortihim
- content based recmonnder - music genome project 
- add regualrtistion to soft max model
- use additional evualtion metrocs
- move onto song reccomenation

Our contribution is to integrate both the content of music objects
and the opinions of the relevant users for better music recommendation. For each user, we
determine the user’s favorite degrees to the music groups in the proposed CB method.
For providing surprising music objects, we further take into account the opinions from
the users of the same user group. The COL method is then proposed for this purpose. In
the MRS, we design a classifier to automatically group the music objects based on the
extracted features from their melodies. This classifier avoids the overhead from the task of
MUSIC RECOMMENDATION SYSTEM BASED ON MUSIC AND USER GROUPING 131
manual classification. Moreover, we derive both the interest and behavior profiles from the
users’ access histories for user grouping. The proposed technique to derive user profiles
has the adaptability to large number of accessed music objects. We also perform a series of
experiments to show that our recommendation system is practical.
The MRS can be regarded as a basic framework. The function blocks, such as the track
selector, the feature extractor, the classifier, and the recommendation module, can be replaced by the alternatives. For example, more features can be extracted for music grouping
by modifying the feature extractor. Similarly, we can adopt different techniques to construct the classifier, such as machine learning or data mining, based on the properties of the
system and the recommendation services. Therefore, our recommendation system is also
flexible.
Our recommendation system can be further enhanced in some ways. For example, to
reduce the time to browse all recommended music objects, the summarization of music
objects may be necessary. Due to the complex semantics contained in the music objects,
new features can be investigated for more effective music grouping. Moreover, other recommendation methods to satisfy various user requirements can be developed

There are several possibilities to extend this work in future. Currently, both heroin and amphetamine addicts are grouped into the same cluster using K-means (even with different numbers of K and principal components).  We plan to experiment with hierarchical clustering algorithms that may be able to model the distinction between these subgroups in a wider ‘poor decision making’ cluster. Furthermore, sub-groups of healthy participants may be revealed with associated advantageous or disadvantageous decision behaviour. In our investigation, we analysed our clusters by study and found little correspondence. However, we would like to analyse our clusters by payoff scheme in an attempt to investigate how participants decision making abilities are impacted by the type of IGT performed. 
In addition, we plan to train a reinforcement learning model on the datasets and perform clustering utilizing the parameters of that model.  Similar endeavours have shown to be fruitful, with such parameters often increasing the interoperability of results. Our devised federated k-means algorithm could also be improved by incorporating mini-batch k-means. This would result in only a trivial reduction in accuracy and would be particularly useful if the decentralized server was a mobile phone. Finally, we are interested in incorporating other features about the subjects such as socio-economic status,  gender, and a chronic gambling addiction indicator. We hope such features might uncover specific card decision patterns or behavioural inabilities during the task.

