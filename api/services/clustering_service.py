"""Service for face clustering operations."""

import logging
import uuid
from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from api.stores import SupabaseVectorStore
from api.schemas.responses import FaceClusterResponse, SampleFace
from api.config import get_settings

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
        min_cluster_size: int = 2,
        on_progress=None
    ) -> ClusterResult:
        """
        Cluster all face embeddings for a user.

        Uses HDBSCAN for density-based clustering.
        on_progress(stage, detail, progress, **kwargs) is called at each step.
        """
        def report(stage, detail, progress, **kwargs):
            logger.info(f"[cluster] {stage}: {detail}")
            if on_progress:
                on_progress(stage, detail, progress, **kwargs)

        # 1. Get all face embeddings for user
        report("Fetching", "Loading face embeddings...", 10)
        embeddings, face_ids = await self._get_all_face_embeddings(user_id)
        report(
            "Fetching",
            f"Found {len(embeddings)} face embeddings",
            15,
            total=len(embeddings)
        )

        if len(embeddings) < min_cluster_size:
            report("Done", f"Only {len(embeddings)} faces found, need at least {min_cluster_size}", 100)
            return ClusterResult(
                num_clusters=0,
                num_faces_clustered=0,
                num_noise_faces=len(embeddings)
            )

        # 2. Auto-label faces from manual asset_tags (skips Apply step)
        report("Auto-label", "Matching faces to manual tags...", 20)
        auto_labeled = await self._auto_label_from_tags(user_id)
        report("Auto-label", f"Labeled {auto_labeled} faces from manual tags", 30)

        # 3. Run clustering
        report(
            "Clustering",
            f"Running HDBSCAN on {len(embeddings)} embeddings "
            f"(threshold={threshold}, min_size={min_cluster_size})...",
            40
        )
        labels = self._run_clustering(
            embeddings,
            threshold,
            min_cluster_size
        )

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = sum(1 for l in labels if l == -1)
        report(
            "Clustering",
            f"Produced {num_clusters} clusters, "
            f"{len(labels) - num_noise} clustered, {num_noise} noise",
            60,
            processed=num_clusters
        )

        # 4. Save cluster assignments
        report("Saving", f"Saving {num_clusters} clusters to database...", 70)
        await self._save_clusters(user_id, face_ids, labels)
        report("Saving", f"Saved {num_clusters} clusters", 95, processed=num_clusters)

        return ClusterResult(
            num_clusters=num_clusters,
            num_faces_clustered=len(labels) - num_noise,
            num_noise_faces=num_noise
        )

    async def _get_all_face_embeddings(
        self,
        user_id: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Get all face embeddings for a user.

        Paginates through Supabase to fetch all rows (default limit is 1000).
        """
        await self._store.connect()

        try:
            all_rows = []
            page_size = 1000
            offset = 0

            while True:
                result = self._store._client.table("face_embeddings") \
                    .select("id, embedding") \
                    .eq("user_id", user_id) \
                    .range(offset, offset + page_size - 1) \
                    .execute()

                if not result.data:
                    break

                all_rows.extend(result.data)

                if len(result.data) < page_size:
                    break
                offset += page_size

            if not all_rows:
                logger.info(f"No face embeddings found for user {user_id}")
                return np.array([]), []

            embeddings = []
            face_ids = []

            for row in all_rows:
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

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(
                f"HDBSCAN produced {num_clusters} clusters from "
                f"{len(embeddings)} embeddings (threshold={threshold}, "
                f"epsilon={min_cluster_dist:.2f})"
            )

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

    @staticmethod
    def _is_normalized(box: dict) -> bool:
        """Check if bounding box uses normalized 0-1 coordinates."""
        return (box.get("x", 0) <= 1 and box.get("y", 0) <= 1 and
                box.get("width", 0) <= 1 and box.get("height", 0) <= 1)

    @staticmethod
    def _normalize_box(box: dict, img_w: int, img_h: int) -> dict:
        """Normalize bounding box to 0-1 range."""
        if not box:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        if ClusteringService._is_normalized(box):
            return {"x": box["x"], "y": box["y"],
                    "width": box["width"], "height": box["height"]}
        w = img_w or 1
        h = img_h or 1
        return {
            "x": box.get("x", 0) / w,
            "y": box.get("y", 0) / h,
            "width": box.get("width", 0) / w,
            "height": box.get("height", 0) / h,
        }

    @staticmethod
    def _calculate_overlap(box1: dict, box2: dict) -> float:
        """Calculate overlap ratio (intersection / smaller area)."""
        x1_1, y1_1 = box1["x"], box1["y"]
        x2_1, y2_1 = x1_1 + box1["width"], y1_1 + box1["height"]
        x1_2, y1_2 = box2["x"], box2["y"]
        x2_2, y2_2 = x1_2 + box2["width"], y1_2 + box2["height"]

        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = box1["width"] * box1["height"]
        a2 = box2["width"] * box2["height"]
        smaller = min(a1, a2)
        return 0.0 if smaller <= 0 else inter / smaller

    async def _auto_label_from_tags(self, user_id: str) -> int:
        """Auto-label face embeddings by cross-referencing manual asset_tags.

        For each unlabeled face, finds overlapping person tags on the same
        asset and inherits contact_id if overlap >= 40%.
        """
        await self._store.connect()

        # Get unlabeled faces with bounding boxes
        faces_result = self._store._client.table("face_embeddings") \
            .select("id, asset_id, bounding_box") \
            .eq("user_id", user_id) \
            .is_("contact_id", "null") \
            .not_.is_("bounding_box", "null") \
            .execute()

        if not faces_result.data:
            return 0

        asset_ids = list(set(f["asset_id"] for f in faces_result.data))

        # Get person tags with contact_id for these assets
        tags_result = self._store._client.table("asset_tags") \
            .select("id, asset_id, contact_id, bounding_box") \
            .eq("created_by", user_id) \
            .eq("tag_type", "person") \
            .not_.is_("contact_id", "null") \
            .in_("asset_id", asset_ids) \
            .execute()

        if not tags_result.data:
            logger.info(f"No person tags with contact_id found for auto-labeling")
            return 0

        # Group tags by asset
        tags_by_asset: dict = {}
        for tag in tags_result.data:
            aid = tag["asset_id"]
            if aid not in tags_by_asset:
                tags_by_asset[aid] = []
            tags_by_asset[aid].append(tag)

        # Get image dimensions for coordinate normalization
        dims_result = self._store._client.table("assets") \
            .select("id, width, height") \
            .in_("id", asset_ids) \
            .execute()

        asset_dims: dict = {}
        for a in (dims_result.data or []):
            asset_dims[a["id"]] = (a.get("width") or 1, a.get("height") or 1)

        # Match faces to tags using bounding box overlap
        labeled = 0
        for face in faces_result.data:
            asset_id = face["asset_id"]
            face_box = face.get("bounding_box")
            if not face_box or asset_id not in tags_by_asset:
                continue

            img_w, img_h = asset_dims.get(asset_id, (1, 1))
            face_norm = self._normalize_box(face_box, img_w, img_h)

            best_contact = None
            best_overlap = 0.0

            for tag in tags_by_asset[asset_id]:
                tag_box = tag.get("bounding_box")
                if not tag_box:
                    continue
                tag_norm = self._normalize_box(tag_box, img_w, img_h)
                overlap = self._calculate_overlap(face_norm, tag_norm)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_contact = tag["contact_id"]

            if best_contact and best_overlap >= 0.4:
                self._store._client.table("face_embeddings") \
                    .update({"contact_id": best_contact}) \
                    .eq("id", face["id"]) \
                    .execute()
                labeled += 1

        logger.info(
            f"Auto-label: {labeled} faces labeled from "
            f"{len(tags_result.data)} manual tags across "
            f"{len(tags_by_asset)} assets"
        )
        return labeled

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

            # Create new clusters, auto-inheriting contact IDs from tagged faces
            auto_assigned = 0
            for label, faces in cluster_faces.items():
                cluster_id = str(uuid.uuid4())

                # Check if any face in this cluster already has a contact_id
                # (from tag sync or previous manual assignment)
                contact_result = self._store._client.table("face_embeddings") \
                    .select("contact_id") \
                    .in_("id", faces) \
                    .not_.is_("contact_id", "null") \
                    .limit(1) \
                    .execute()

                inherited_contact_id = None
                if contact_result.data and len(contact_result.data) > 0:
                    inherited_contact_id = contact_result.data[0]["contact_id"]

                # Create cluster record
                cluster_record = {
                    "id": cluster_id,
                    "user_id": user_id,
                    "face_count": len(faces),
                    "representative_face_id": faces[0] if faces else None,
                }
                if inherited_contact_id:
                    cluster_record["contact_id"] = inherited_contact_id
                    auto_assigned += 1

                self._store._client.table("face_clusters").insert(
                    cluster_record
                ).execute()

                # Update face_embeddings with cluster_id (and contact if inherited)
                update_data = {"cluster_id": cluster_id}
                if inherited_contact_id:
                    update_data["contact_id"] = inherited_contact_id

                for face_id in faces:
                    self._store._client.table("face_embeddings") \
                        .update(update_data) \
                        .eq("id", face_id) \
                        .execute()

            logger.info(
                f"Saved {len(cluster_faces)} clusters, "
                f"{auto_assigned} auto-assigned from tag sync labels"
            )

        except Exception as e:
            logger.error(f"Failed to save clusters: {e}")

    async def get_clusters(
        self,
        user_id: str
    ) -> List[FaceClusterResponse]:
        """Get all face clusters for a user."""
        await self._store.connect()

        try:
            # Paginate to fetch all clusters (Supabase defaults to 1000 rows)
            all_rows = []
            page_size = 1000
            offset = 0

            while True:
                result = self._store._client.table("face_clusters") \
                    .select("*") \
                    .eq("user_id", user_id) \
                    .order("face_count", desc=True) \
                    .range(offset, offset + page_size - 1) \
                    .execute()

                if not result.data:
                    break
                all_rows.extend(result.data)
                if len(result.data) < page_size:
                    break
                offset += page_size

            if not all_rows:
                logger.info(f"No clusters found for user {user_id}")
                return []

            # Batch-fetch one sample face per cluster using a single query
            cluster_ids = [row["id"] for row in all_rows]
            sample_faces_map = await self._get_sample_faces_batch(cluster_ids)

            clusters = []
            for row in all_rows:
                cluster = FaceClusterResponse(
                    cluster_id=row["id"],
                    name=row.get("name"),
                    contact_id=row.get("contact_id"),
                    face_count=row.get("face_count", 0),
                    sample_faces=sample_faces_map.get(row["id"], [])
                )
                clusters.append(cluster)

            logger.info(f"Found {len(clusters)} clusters for user {user_id}")
            return clusters

        except Exception as e:
            logger.error(f"Failed to get clusters: {e}")
            return []

    async def _get_sample_faces_batch(
        self,
        cluster_ids: List[str]
    ) -> dict:
        """Get one sample face per cluster in bulk.

        Returns {cluster_id: [SampleFace, ...]} with one face each.
        Uses representative_face_id from face_embeddings in a single query.
        """
        if not cluster_ids:
            return {}

        try:
            # Fetch representative faces for all clusters in one paginated query
            all_faces = []
            page_size = 1000
            offset = 0

            while True:
                batch = cluster_ids[offset:offset + page_size]
                if not batch:
                    break

                result = self._store._client.table("face_embeddings") \
                    .select("cluster_id, asset_id, face_index, is_from_video") \
                    .in_("cluster_id", batch) \
                    .execute()

                if result.data:
                    all_faces.extend(result.data)

                if len(batch) < page_size:
                    break
                offset += page_size

            # Group by cluster, keep first face per cluster
            # (prefer faces with thumbnails)
            faces_by_cluster: dict = {}
            for row in all_faces:
                cid = row["cluster_id"]
                if cid not in faces_by_cluster:
                    faces_by_cluster[cid] = []
                faces_by_cluster[cid].append(row)

            result_map = {}
            all_asset_ids = set()
            for cid, faces in faces_by_cluster.items():
                best = faces[0] if faces else None
                if best:
                    result_map[cid] = {
                        "asset_id": best["asset_id"],
                        "face_index": best.get("face_index", 0),
                        "thumbnail_url": None,
                        "is_from_video": best.get("is_from_video", False),
                    }
                    all_asset_ids.add(best["asset_id"])

            # Batch-fetch asset thumbnails
            if all_asset_ids:
                asset_result = self._store._client.table("album_assets") \
                    .select("asset_id, asset_uri, thumbnail_uri, asset_type") \
                    .in_("asset_id", list(all_asset_ids)) \
                    .execute()

                asset_urls: dict = {}
                if asset_result.data:
                    for row in asset_result.data:
                        aid = row["asset_id"]
                        thumb = row.get("thumbnail_uri")
                        uri = row.get("asset_uri")
                        atype = row.get("asset_type", "photo")
                        if thumb and thumb.startswith("http"):
                            asset_urls[aid] = thumb
                        elif atype == "photo" and uri and uri.startswith("http"):
                            asset_urls[aid] = uri
                        elif uri and uri.startswith("http"):
                            lower = uri.lower()
                            if any(lower.endswith(e) for e in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                                asset_urls[aid] = uri

                for cid, data in result_map.items():
                    data["thumbnail_url"] = asset_urls.get(data["asset_id"])

            # Convert to SampleFace objects
            final_map = {}
            for cid, data in result_map.items():
                final_map[cid] = [SampleFace(
                    asset_id=data["asset_id"],
                    face_index=data["face_index"],
                    thumbnail_url=data["thumbnail_url"],
                    is_from_video=data["is_from_video"],
                )]

            return final_map

        except Exception as e:
            logger.error(f"Failed to batch-fetch sample faces: {e}")
            return {}

    async def _get_sample_faces(
        self,
        cluster_id: str,
        limit: int = 4
    ) -> List[SampleFace]:
        """Get sample faces for a cluster with asset thumbnail URLs and bounding boxes."""
        try:
            result = self._store._client.table("face_embeddings") \
                .select("asset_id, face_index, bounding_box, is_from_video") \
                .eq("cluster_id", cluster_id) \
                .limit(limit * 3) \
                .execute()

            if not result.data:
                return []

            faces = []
            for row in result.data:
                asset_id = row["asset_id"]
                thumbnail_url = await self._get_asset_thumbnail(asset_id)

                faces.append(SampleFace(
                    asset_id=asset_id,
                    face_index=row.get("face_index", 0),
                    thumbnail_url=thumbnail_url,
                    bounding_box=row.get("bounding_box"),
                    is_from_video=row.get("is_from_video", False),
                ))

            # Prioritize faces with thumbnails
            with_thumbs = [f for f in faces if f.thumbnail_url]
            without_thumbs = [f for f in faces if not f.thumbnail_url]
            sample_faces = with_thumbs[:limit]
            remaining = limit - len(sample_faces)
            if remaining > 0:
                sample_faces.extend(without_thumbs[:remaining])

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
            result = self._store._client.table("face_embeddings") \
                .select("id, asset_id, face_index, bounding_box, is_from_video") \
                .eq("cluster_id", cluster_id) \
                .eq("user_id", user_id) \
                .execute()

            if not result.data:
                return []

            # Batch-fetch asset thumbnails
            all_asset_ids = list(set(row["asset_id"] for row in result.data))
            asset_urls: dict = {}
            asset_result = self._store._client.table("album_assets") \
                .select("asset_id, asset_uri, thumbnail_uri, asset_type") \
                .in_("asset_id", all_asset_ids) \
                .execute()
            if asset_result.data:
                for arow in asset_result.data:
                    aid = arow["asset_id"]
                    thumb = arow.get("thumbnail_uri")
                    uri = arow.get("asset_uri")
                    atype = arow.get("asset_type", "photo")
                    if thumb and thumb.startswith("http"):
                        asset_urls[aid] = thumb
                    elif atype == "photo" and uri and uri.startswith("http"):
                        asset_urls[aid] = uri
                    elif uri and uri.startswith("http"):
                        lower = uri.lower()
                        if any(lower.endswith(e) for e in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                            asset_urls[aid] = uri

            faces = []
            for row in result.data:
                asset_id = row["asset_id"]
                faces.append({
                    "id": row["id"],
                    "asset_id": asset_id,
                    "face_index": row.get("face_index", 0),
                    "thumbnail_url": asset_urls.get(asset_id),
                    "is_from_video": row.get("is_from_video", False),
                    "bounding_box": row.get("bounding_box")
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
            update_data = {"contact_id": contact_id}
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
                            .update({"cluster_id": None, "contact_id": None}) \
                            .eq("id", face_id) \
                            .execute()
                    else:
                        # Assign to contact
                        self._store._client.table("face_embeddings") \
                            .update({"contact_id": contact_id}) \
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

    async def recognize_faces(
        self,
        asset_id: str,
        user_id: str,
        similarity_threshold: float = 0.55,
        worker_id: str = None
    ) -> int:
        """Recognize faces in a newly processed image by comparing against
        labeled embeddings. Assigns contact_id and cluster_id to
        any matching faces.

        Called after face detection during image processing.

        Returns number of faces recognized.
        """
        await self._store.connect()
        client = self._store._client

        try:
            # Get new (unlabeled) face embeddings for this asset
            new_faces = client.table("face_embeddings") \
                .select("id, embedding, bounding_box, face_index") \
                .eq("asset_id", asset_id) \
                .eq("user_id", user_id) \
                .is_("contact_id", "null") \
                .execute()

            if not new_faces.data:
                return 0

            # Get all labeled face embeddings for this user
            labeled = client.table("face_embeddings") \
                .select("id, embedding, contact_id, cluster_id") \
                .eq("user_id", user_id) \
                .not_.is_("contact_id", "null") \
                .execute()

            if not labeled.data:
                logger.debug(f"[recognize] No labeled faces for user {user_id}")
                return 0

            # Parse labeled embeddings once
            labeled_vecs = []
            labeled_meta = []
            for row in labeled.data:
                vec = parse_pgvector(row.get("embedding", ""))
                if vec:
                    labeled_vecs.append(vec)
                    labeled_meta.append({
                        "contact_id": row["contact_id"],
                        "cluster_id": row.get("cluster_id"),
                    })

            if not labeled_vecs:
                return 0

            # Build contact name cache for logging
            contact_ids = list(set(
                m["contact_id"] for m in labeled_meta
            ))
            contact_names: dict = {}
            if contact_ids:
                names_result = client.table("contacts") \
                    .select("id, display_name") \
                    .in_("id", contact_ids) \
                    .execute()
                for c in (names_result.data or []):
                    contact_names[c["id"]] = c.get("display_name") or "Unknown"

            labeled_matrix = np.array(labeled_vecs)
            # Pre-normalize for cosine similarity
            labeled_norms = np.linalg.norm(labeled_matrix, axis=1, keepdims=True)
            labeled_normed = labeled_matrix / np.maximum(labeled_norms, 1e-10)

            recognized = 0
            for face_row in new_faces.data:
                face_vec = parse_pgvector(face_row.get("embedding", ""))
                if not face_vec:
                    continue

                face_arr = np.array(face_vec)
                face_norm = np.linalg.norm(face_arr)
                if face_norm < 1e-10:
                    continue
                face_normed = face_arr / face_norm

                # Cosine similarity against all labeled faces
                similarities = labeled_normed @ face_normed
                best_idx = int(np.argmax(similarities))
                best_sim = float(similarities[best_idx])

                if best_sim >= similarity_threshold:
                    match = labeled_meta[best_idx]
                    update = {"contact_id": match["contact_id"]}
                    if match.get("cluster_id"):
                        update["cluster_id"] = match["cluster_id"]

                    client.table("face_embeddings") \
                        .update(update) \
                        .eq("id", face_row["id"]) \
                        .execute()

                    # Update cluster face count if assigned to cluster
                    if match.get("cluster_id"):
                        count_result = client.table("face_embeddings") \
                            .select("id", count="exact") \
                            .eq("cluster_id", match["cluster_id"]) \
                            .execute()
                        client.table("face_clusters") \
                            .update({"face_count": count_result.count or 0}) \
                            .eq("id", match["cluster_id"]) \
                            .execute()

                    # Create a person tag in asset_tags so the UI shows it
                    cid = match["contact_id"]
                    cname = contact_names.get(cid, cid[:8])
                    try:
                        client.table("asset_tags").insert({
                            "id": str(uuid.uuid4()),
                            "asset_id": asset_id,
                            "tag_type": "person",
                            "tag_value": cname,
                            "contact_id": cid,
                            "bounding_box": face_row.get("bounding_box"),
                            "face_index": face_row.get("face_index"),
                            "created_by": user_id,
                            "tagged_by": {
                                "source": "kizu",
                                "worker_id": worker_id or "unknown",
                                "model": f"{get_settings().default_face_model}-{get_settings().face_model_name}",
                                "similarity": round(best_sim, 3),
                            },
                        }).execute()
                    except Exception as tag_err:
                        logger.warning(
                            f"[recognize] Failed to create tag: {tag_err}"
                        )

                    recognized += 1
                    logger.info(
                        f"[recognize] Face {face_row['id'][:8]} in "
                        f"asset {asset_id[:8]} → \"{cname}\" "
                        f"(sim={best_sim:.3f}, cluster={match.get('cluster_id', 'none')[:8] if match.get('cluster_id') else 'none'})"
                    )

            if recognized > 0:
                logger.info(
                    f"[recognize] Recognized {recognized}/{len(new_faces.data)} "
                    f"faces in asset {asset_id[:8]}"
                )
            return recognized

        except Exception as e:
            logger.error(f"[recognize] Failed for asset {asset_id}: {e}")
            return 0
