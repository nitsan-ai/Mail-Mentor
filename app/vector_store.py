import logging
import numpy as np
from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


class MilvusVectorStore:
    def __init__(self, collection_name="emails", vector_dim=768, db_file="milvus_emails.db"):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.db_file = db_file
        self.client = self._initialize_client()
        self._create_collection_fresh()

    def _initialize_client(self):
        try:
            client = MilvusClient(self.db_file)
            logger.info(f"✅ Initialized Milvus Lite client with database: {self.db_file}")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to initialize Milvus client: {e}")
            raise

    def _create_collection_fresh(self):
        if self.client.has_collection(self.collection_name):
            logger.warning(f"Dropping existing collection '{self.collection_name}' for fresh start...")
            self.client.drop_collection(self.collection_name)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_fields=True)

        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=255
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.vector_dim
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.VARCHAR,
            max_length=65535
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_name="embedding_index",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"✅ Created fresh collection '{self.collection_name}' with metadata support")

    def get_existing_ids(self, ids: List[str]) -> List[str]:
        if not ids:
            return []
        try:
            results = self.client.get(collection_name=self.collection_name, ids=ids)
            return [res['id'] for res in results] if results else []
        except Exception as e:
            logger.error(f"❌ Failed to fetch existing IDs: {e}")
            return []

    def add_emails(self, emails: List[Dict[str, Any]], embeddings: np.ndarray):
        if not emails or embeddings is None or len(emails) == 0:
            logger.warning("⚠️ No emails to insert, skipping.")
            return

        logger.info(f"📥 Ingesting {len(emails)} emails into Milvus...")

        data_to_insert = []
        for email, embedding in zip(emails, embeddings):
            data_to_insert.append({
                "id": email['id'],
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "metadata": json.dumps(email)
            })

        try:
            self.client.upsert(collection_name=self.collection_name, data=data_to_insert)
            self.client.flush(self.collection_name)
            logger.info(f"✅ Successfully ingested and flushed {len(emails)} emails.")
        except Exception as e:
            logger.error(f"❌ Failed to insert emails: {e}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            if stats.get("row_count", 0) == 0:
                logger.warning("⚠️ Search attempted on empty collection. Returning no results.")
                return []
        except Exception as e:
            logger.error(f"❌ Could not get collection stats: {e}")
            return []

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding.tolist()],
                anns_field="embedding",
                limit=top_k,
                search_params=search_params,
                output_fields=["metadata"]
            )
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

        formatted_results = []
        if results and results[0]:
            for hit in results[0]:
                try:
                    email_data = json.loads(hit['entity']['metadata'])
                    email_data['score'] = hit['distance']
                    formatted_results.append(email_data)
                except Exception as e:
                    logger.error(f"Failed to parse metadata from hit: {e}")

        return formatted_results