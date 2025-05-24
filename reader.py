from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

cred_path = "key.json"
credentials = service_account.Credentials.from_service_account_file(cred_path)

project_id = 'poetic-now-460508-p0'
client = bigquery.Client(credentials= credentials,project=project_id)

query_job = client.query("""
   SELECT user_pseudo_id AS userId,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS pageURL,
  COUNT(*) AS rating FROM
  bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131
  GROUP BY userId, pageURL
 HAVING pageURL LIKE '%shop.googlemerchandisestore.com%'
""")

results = query_job.result() # Wait for the job to complete.
data = query_job.to_dataframe()
print(data.head(2))


data.to_csv('train_data.csv', index=False)
