# Text Similarity Computing Based on Spark.MLlib's LDA

### Overview
Text similarity computing based on traditional TF-IDF model exists the problem of high dimensional sparse data and lack of semantics. Here it proposed a text similarity calculation method by using Spark.MLlib's LDA (Latent Dirichlet Allocation), distributed LDA model can solve all these problems, particularly in handling voluminous chunks of data. After generating distribution over topics of documents, use the KL (Kullback-Leibler divergence) distance computing text similarity. The experiment result shows that the proposed method can efficiently recognize the similar texts.

### Description 
![image](https://github.com/yangym6/text-similarity-spark-lda/blob/master/screenshots/sim_1.png)

![image](https://github.com/yangym6/text-similarity-spark-lda/blob/master/screenshots/sim_2.png)

![image](https://github.com/yangym6/text-similarity-spark-lda/blob/master/screenshots/sim_3.png)
 
![image](https://github.com/yangym6/text-similarity-spark-lda/blob/master/screenshots/sim_4.png)
