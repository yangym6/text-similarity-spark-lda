# text-similarity-spark-lda
Text Similarity Computing Based on Spark.MLlib's LDA

Text similarity computing based on traditional TF-IDF model exists the problem of high dimensional sparse data and lack of semantics. Here it proposed a text similarity calculation method by using Spark.MLlib's LDA (Latent Dirichlet Allocation), distributed LDA model can solve all these problems, particularly in handling voluminous chunks of data. And then use KL (Kullback-Leibler divergence) distance computing text similarity. The experiment result shows that the proposed method can efficiently recognize the similar texts.
