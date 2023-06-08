from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define the categories/topics you want to visualize
categories_6 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                'rec.sport.baseball', 'rec.sport.hockey', 'sci.space']

# Fetch the data for the specified categories
groups_6 = fetch_20newsgroups(categories=categories_6)

# Preprocess the data
count_vector_sw = CountVectorizer(stop_words="english", max_features=5000)
data_cleaned_count_6 = count_vector_sw.fit_transform(groups_6.data)

# Perform t-SNE dimensionality reduction
tsne_model = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
data_tsne = tsne_model.fit_transform(data_cleaned_count_6.toarray())

# Visualize the data
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_6.target)
plt.show()
