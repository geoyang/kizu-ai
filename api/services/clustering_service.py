"""Service for face clustering operations."""

import logging
import uuid
from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from api.stores import SupabaseVectorStore
from api.schemas.responses import FaceClusterResponse, SampleFace

logger = logging.getLogger(__name__)


def parse_pgvector(vector_str: str) -> List[float]:
    """Parse pgvector string format [x,y,z] to list of floats."""
    if not vector_str:
        return []
    # Handle string format: "[0.1,0.2,0.3]"
    if isinstance(vector_str, str):
        cleaned = vector_str.strip('[]')
        if not cleaned:
            return []
        return [float(x) for x in cleaned.split(',')]
    # Handle list format from some drivers
    return list(vector_str)


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    num_clusters: int
    num_faces_clustered: int
    num_noise_faces: int


class ClusteringService:
    """Service for clustering face embeddings."""

    def __init__(self, vector_store: SupabaseVectorStore):
        self._store = vector_store

    async def cluster_faces(
        self,
        user_id: str,
        threshold: float = 0.6,
        min_cluster_size: int = 2
    ) -> ClusterResult:
        """
        Cluster all face embeddings for a user.

        Uses HDBSCAN for density-based clustering.
        """
        # 1. Get all face embeddings for user
        embeddings, face_ids = await self._get_all_face_embeddings(user_id)

        if len(embeddings) < min_cluster_size:
            return ClusterResult(
                num_clusters=0,
                num_faces_clustered=0,
                num_noise_faces=len(embeddings)
            )

        # 2. Run clustering
        labels = self._run_clustering(
            embeddings,
            threshold,
            min_cluster_size
        )

        # 3. Save cluster assignments
        await self._save_clusters(user_id, face_ids, labels)

        # 4. Return stats
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = sum(1 for l in labels if l == -1)

        return ClusterResult(
            num_clusters=num_clusters,
            num_faces_clustered=len(labels) - num_noise,
            num_noise_faces=num_noise
        )

    async def _get_all_face_embeddings(
        self,
        user_id: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Get all face embeddings for a user."""
        await self._store.connect()

        try:
            result = self._store._client.table("face_embeddings") \
                .select("id, embedding") \
                .eq("user_id", user_id) \
                .execute()

            if not result.data:
                logger.info(f"No face embeddings found for user {user_id}")
                return np.array([]), []

            embeddings = []
            face_ids = []

            for row in result.data:
                embedding_data = row.get("embedding")
                if embedding_data:
                    parsed = parse_pgvector(embedding_data)
                    if parsed:
                        embeddings.append(parsed)
                        face_ids.append(row["id"])

            logger.info(f"Found {len(embeddings)} face embeddings for user {user_id}")
            return np.array(embeddings), face_ids

        except Exception as e:
            logger.error(f"Failed to get face embeddings: {e}")
            return np.array([]), []

    def _run_clustering(
        self,
        embeddings: np.ndarray,
        threshold: float,
        min_cluster_size: int
    ) -> List[int]:
        """Run HDBSCAN clustering on embeddings."""
        try:
            import hdbscan

            # Convert threshold to distance (1 - similarity)
            min_cluster_dist = 1 - threshold

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                metric='euclidean',
                cluster_selection_epsilon=min_cluster_dist
            )

            labels = clusterer.fit_predict(embeddings)
            return labels.tolist()

        except ImportError:
            logger.warning("HDBSCAN not available, using simple clustering")
            return self._simple_clustering(embeddings, threshold)

    def _simple_clustering(
        self,
        embeddings: np.ndarray,
        threshold: float
    ) -> List[int]:
        """Simple greedy clustering fallback."""
        if len(embeddings) == 0:
            return []

        labels = [-1] * len(embeddings)
        current_cluster = 0

        for i in range(len(embeddings)):
            if labels[i] != -1:
                continue

            labels[i] = current_cluster
            cluster_center = embeddings[i]

            for j in range(i + 1, len(embeddings)):
                if labels[j] != -1:
                    continue

                similarity = np.dot(cluster_center, embeddings[j]) / (
                    np.linalg.norm(cluster_center) * np.linalg.norm(embeddings[j])
                )

                if similarity >= threshold:
                    labels[j] = current_cluster

            current_cluster += 1

        return labels

    async def _save_clusters(
        self,
        user_id: str,
        face_ids: List[str],
        labels: List[int]
    ) -> None:
        """Save cluster assignments to database."""
        await self._store.connect()

        try:
            # Group faces by cluster label
            cluster_faces = {}
            for face_id, label in zip(face_ids, labels):
                if label == -1:  # Noise/unclustered
                    continue
                if label not in cluster_faces:
                    cluster_faces[label] = []
                cluster_faces[label].append(face_id)

            logger.info(f"Creating {len(cluster_faces)} clusters for user {user_id}")

            # Delete existing clusters for this user (fresh clustering)
            self._store._client.table("face_clusters") \
                .delete() \
                .eq("user_id", user_id) \
                .execute()

            # Clear cluster_id from all face_embeddings for this user
            self._store._client.table("face_embeddings") \
                .update({"cluster_id": None}) \
                .eq("user_id", user_id) \
                .execute()

            # Create new clusters
            for label, faces in cluster_faces.items():
                cluster_id = str(uuid.uuid4())

                # Create cluster record
                self._store._client.table("face_clusters").insert({
                    "id": cluster_id,
                    "user_id": user_id,
                    "face_count": len(faces),
                    "representative_face_id": faces[0] if faces else None
                }).execute()

                # Update face_embeddings with cluster_id
                for face_id in faces:
                    self._store._client.table("face_embeddings") \
                        .update({"cluster_id": cluster_id}) \
                        .eq("id", face_id) \
                        .execute()

            logger.info(f"Successfully saved {len(cluster_faces)} clusters")

        except Exception as e:
            logger.error(f"Failed to save clusters: {e}")

    async def get_clusters(
        self,
        user_id: str
    ) -> List[FaceClusterResponse]:
        """Get all face clusters for a user."""
        await self._store.connect()

        try:
            result = self._store._client.table("face_clusters") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("face_count", desc=True) \
                .execute()

            if not result.data:
                logger.info(f"No clusters found for user {user_id}")
                return []

            clusters = []
            for row in result.data:
                # Get sample faces for this cluster
                sample_faces = await self._get_sample_faces(row["id"], limit=4)

                cluster = FaceClusterResponse(
                    cluster_id=row["id"],
                    name=row.get("name"),
                    knox_contact_id=row.get("knox_contact_id"),
                    face_count=row.get("face_count", 0),
                    sample_faces=sample_faces
                )
                clusters.append(cluster)

            logger.info(f"Found {len(clusters)} clusters for user {user_id}")
            return clusters

        except Exception as e:
            logger.error(f"Failed to get clusters: {e}")
            return []

    async def _get_sample_faces(
        self,
        cluster_id: str,
        limit: int = 4
    ) -> List[SampleFace]:
        """Get sample faces for a cluster with thumbnail URLs.

        Uses face_thumbnail_url from face_embeddings if available,
        otherwise falls back to album_assets for legacy data.
        """
        try:
            # Get face embeddings with thumbnail URLs
            result = self._store._client.table("face_embeddings") \
                .select("asset_id, face_index, bounding_box, face_thumbnail_url, is_from_video") \
                .eq("cluster_id", cluster_id) \
                .limit(limit * 3) \
                .execute()

            if not result.data:
                return []

            # Separate faces with and without thumbnails
            faces_with_thumbnails = []
            faces_without_thumbnails = []

            for row in result.data:
                asset_id = row["asset_id"]
                thumbnail_url = row.get("face_thumbnail_url")
                is_from_video = row.get("is_from_video", False)

                # If no face_thumbnail_url, try to get from album_assets (legacy fallback)
                if not thumbnail_url:
                    thumbnail_url = await self._get_asset_thumbnail(asset_id)

                face = SampleFace(
                    asset_id=asset_id,
                    face_index=row.get("face_index", 0),
                    thumbnail_url=thumbnail_url,
                    is_from_video=is_from_video
                )

                if thumbnail_url:
                    faces_with_thumbnails.append(face)
                else:
                    faces_without_thumbnails.append(face)

            # Prioritize faces with thumbnails
            sample_faces = faces_with_thumbnails[:limit]
            remaining = limit - len(sample_faces)
            if remaining > 0:
                sample_faces.extend(faces_without_thumbnails[:remaining])

            return sample_faces

        except Exception as e:
            logger.error(f"Failed to get sample faces: {e}")
            return []

    async def _get_asset_thumbnail(self, asset_id: str) -> Optional[str]:
        """Get thumbnail URL from album_assets table (legacy fallback)."""
        try:
            asset_result = self._store._client.table("album_assets") \
                .select("asset_uri, thumbnail_uri, asset_type") \
                .eq("asset_id", asset_id) \
                .limit(1) \
                .execute()

            if not asset_result.data or len(asset_result.data) == 0:
                return None

            row_data = asset_result.data[0]
            thumbnail_uri = row_data.get("thumbnail_uri")
            asset_uri = row_data.get("asset_uri")
            asset_type = row_data.get("asset_type", "photo")

            # Use thumbnail_uri if available
            if thumbnail_uri and thumbnail_uri.startswith("http"):
                return thumbnail_uri

            # For photos, use asset_uri directly
            if asset_type == "photo" and asset_uri and asset_uri.startswith("http"):
                return asset_uri

            # For videos, only use if it's an image format
            if asset_uri and asset_uri.startswith("http"):
                lower_uri = asset_uri.lower()
                if any(lower_uri.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                    return asset_uri

            return None

        except Exception as e:
            logger.error(f"Failed to get asset thumbnail for {asset_id}: {e}")
            return None

    async def get_all_faces_for_cluster(
        self,
        cluster_id: str,
        user_id: str
    ) -> List[dict]:
        """Get ALL faces for a cluster (not just samples)."""
        await self._store.connect()

        try:
            # Get all face embeddings for this cluster (include id for selection)
            result = self._store._client.table("face_embeddings") \
                .select("id, asset_id, face_index, bounding_box, face_thumbnail_url, is_from_video") \
                .eq("cluster_id", cluster_id) \
                .eq("user_id", user_id) \
                .execute()

            if not result.data:
                return []

            faces = []
            for row in result.data:
                asset_id = row["asset_id"]
                thumbnail_url = row.get("face_thumbnail_url")
                bounding_box = row.get("bounding_box")

                # If no face_thumbnail_url, try to get from album_assets (legacy fallback)
                if not thumbnail_url:
                    thumbnail_url = await self._get_asset_thumbnail(asset_id)

                faces.append({
                    "id": row["id"],  # Face embedding ID for selection
                    "asset_id": asset_id,
                    "face_index": row.get("face_index", 0),
                    "thumbnail_url": thumbnail_url,
                    "is_from_video": row.get("is_from_video", False),
                    "bounding_box": bounding_box
                })

            logger.info(f"Cluster {cluster_id}: returning {len(faces)} total faces")
            return faces

        except Exception as e:
            logger.error(f"Failed to get all faces for cluster: {e}")
            return []

    async def assign_cluster_to_contact(
        self,
        cluster_id: str,
        contact_id: str,
        name: Optional[str],
        user_id: str,
        exclude_face_ids: Optional[List[str]] = None
    ) -> bool:
        """Assign a cluster to a Knox contact, optionally excluding certain faces."""
        await self._store.connect()

        try:
            # Update face_clusters with contact assignment
            update_data = {"knox_contact_id": contact_id}
            if name:
                update_data["name"] = name

            self._store._client.table("face_clusters") \
                .update(update_data) \
                .eq("id", cluster_id) \
                .eq("user_id", user_id) \
                .execute()

            # Get all face embeddings in this cluster
            faces_result = self._store._client.table("face_embeddings") \
                .select("id") \
                .eq("cluster_id", cluster_id) \
                .execute()

            if faces_result.data:
                for face in faces_result.data:
                    face_id = face["id"]
                    if exclude_face_ids and face_id in exclude_face_ids:
                        # Remove excluded faces from cluster (set cluster_id to null)
                        self._store._client.table("face_embeddings") \
                            .update({"cluster_id": None, "knox_contact_id": None}) \
                            .eq("id", face_id) \
                            .execute()
                    else:
                        # Assign to contact
                        self._store._client.table("face_embeddings") \
                            .update({"knox_contact_id": contact_id}) \
                            .eq("id", face_id) \
                            .execute()

            # Update cluster face count
            remaining_count = self._store._client.table("face_embeddings") \
                .select("id", count="exact") \
                .eq("cluster_id", cluster_id) \
                .execute()

            self._store._client.table("face_clusters") \
                .update({"face_count": remaining_count.count or 0}) \
                .eq("id", cluster_id) \
                .execute()

            logger.info(f"Assigned cluster {cluster_id} to contact {contact_id}, excluded {len(exclude_face_ids or [])} faces")
            return True

        except Exception as e:
            logger.error(f"Failed to assign cluster: {e}")
            return False

    async def remove_faces_from_cluster(
        self,
        cluster_id: str,
        face_ids: List[str],
        user_id: str
    ) -> bool:
        """Remove specific faces from a cluster."""
        await self._store.connect()

        try:
            for face_id in face_ids:
                self._store._client.table("face_embeddings") \
                    .update({"cluster_id": None}) \
                    .eq("id", face_id) \
                    .eq("user_id", user_id) \
                    .execute()

            # Update cluster face count
            remaining_count = self._store._client.table("face_embeddings") \
                .select("id", count="exact") \
                .eq("cluster_id", cluster_id) \
                .execute()

            self._store._client.table("face_clusters") \
                .update({"face_count": remaining_count.count or 0}) \
                .eq("id", cluster_id) \
                .execute()

            logger.info(f"Removed {len(face_ids)} faces from cluster {cluster_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove faces from cluster: {e}")
            return False

    async def merge_clusters(
        self,
        cluster_ids: List[str],
        user_id: str
    ) -> Optional[str]:
        """Merge multiple clusters into one."""
        await self._store.connect()

        if len(cluster_ids) < 2:
            return None

        try:
            # Use the first cluster as the target
            target_cluster_id = cluster_ids[0]
            source_cluster_ids = cluster_ids[1:]

            # Get total face count
            total_faces = 0
            for cid in cluster_ids:
                result = self._store._client.table("face_embeddings") \
                    .select("id", count="exact") \
                    .eq("cluster_id", cid) \
                    .execute()
                total_faces += result.count or 0

            # Move all faces to target cluster
            for source_id in source_cluster_ids:
                self._store._client.table("face_embeddings") \
                    .update({"cluster_id": target_cluster_id}) \
                    .eq("cluster_id", source_id) \
                    .execute()

                # Delete the source cluster
                self._store._client.table("face_clusters") \
                    .delete() \
                    .eq("id", source_id) \
                    .execute()

            # Update target cluster face count
            self._store._client.table("face_clusters") \
                .update({"face_count": total_faces}) \
                .eq("id", target_cluster_id) \
                .execute()

            logger.info(f"Merged {len(cluster_ids)} clusters into {target_cluster_id}")
            return target_cluster_id

        except Exception as e:
            logger.error(f"Failed to merge clusters: {e}")
            return None
