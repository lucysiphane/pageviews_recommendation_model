
# USER PAGE RECOMMENDATION SYSTEM USING MATRIX FACTORIZATION

The project implements a user page recommendation system using matrix factorization. 
The system retrieves user-page interaction data from BigQuery, processes it, and generates personalized recommendations based on user ratings.

## Overview
- **Data Source**: Google BigQuery
- **Technique**: Matrix Factorization
- **Similarity Metric**: Cosine Similarity
- **Language**: Python
- **Libraries**: Pandas, NumPy, SciPy, Scikit-learn

## Data Retrieval
Data is retrieved directly from **Google BigQuery**. 
```query_job = client.query("""
   SELECT user_pseudo_id AS userId,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS pageURL,
  COUNT(*) AS rating FROM
  bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131
  GROUP BY userId, pageURL
 HAVING pageURL LIKE '%shop.googlemerchandisestore.com%'
""")```
#### Read the data as cvs
convert userId from float to string
```data=pd.read_csv('train_data.csv')
data
data['userId'] = data['userId'].astype('string')
data.head(5)```

## Data Preprocessing
Encode user_id and page_url
user_encoder = LabelEncoder()
page_encoder = LabelEncoder()
```data['encoded_userId'] = user_encoder.fit_transform(data['userId'])
data['encoded_pageURL'] = page_encoder.fit_transform(data['pageURL'])
data['rating'] = pd.to_numeric(data['rating'], errors='coerce').fillna(0).astype(float)
data.head(3)```

## Sparse Matrix Construction
construct a user-item sparse matrix to efficiently represent the large, sparse dataset.
```ratings_matrix = csr_matrix(
    (data['rating'], (data['encoded_userId'], data['encoded_pageURL'])),
    shape=(len(user_encoder.classes_), len(page_encoder.classes_))
)```

## Similarity Calculation
Using cosine similarity, compute how similar items are based on user interactions
```item_similarity = cosine_similarity(ratings_matrix.T, dense_output=False)```

## Recommendation Function
It gets recommendations for a user by comparing their interaction profile with the item similarity matrix.
```def recommend_items(user_id, item_similarity, user_item_matrix, top_n=5):
    encoded_user_id = user_encoder.transform([user_id])[0]
    user_interactions = user_item_matrix[encoded_user_id, :]
    scores = item_similarity.dot(user_interactions.T).toarray().ravel()
    known_items = user_interactions.nonzero()[1]
    scores[known_items] = -1
    recommended_item_indices = np.argsort(scores)[::-1][:top_n]
    recommended_items = page_encoder.inverse_transform(recommended_item_indices)
    return recommended_items```

## Testing the System
Test the recommendation system with a specific user ID to see the output in action.
```print("Recommend pages:", recommend_items('1026454.4271112503', item_similarity, ratings_matrix))```