from qdrant_client import QdrantClient, models
from qdrant_client import QdrantClient

# COLLECTION_NAME = "ageye"

client = QdrantClient(
    url="https://d5ecbd2d-10c9-4d63-baa9-758083e15268.eu-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ot8jDlyrSXQjPctfOcyK1q5oolQ_wKAabsCpfUPQXc4",
)
print(client.get_collections())