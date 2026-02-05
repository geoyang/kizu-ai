"""Service for generating photo moments based on location, people, events, and dates."""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MomentCandidate:
    """A candidate moment before final selection."""
    grouping_type: str
    grouping_criteria: Dict
    asset_ids: List[str]
    title: str
    subtitle: Optional[str]
    date_range_start: Optional[datetime]
    date_range_end: Optional[datetime]
    score: float = 0.0


@dataclass
class GeneratedMoment:
    """A fully generated moment ready for database insertion."""
    grouping_type: str
    grouping_criteria: Dict
    title: str
    subtitle: Optional[str]
    cover_asset_ids: List[str]
    all_asset_ids: List[str]
    date_range_start: Optional[datetime]
    date_range_end: Optional[datetime]


def parse_pgvector(vector_str: str) -> List[float]:
    """Parse pgvector string format [x,y,z] to list of floats."""
    if not vector_str:
        return []
    if isinstance(vector_str, str):
        cleaned = vector_str.strip('[]')
        if not cleaned:
            return []
        return [float(x) for x in cleaned.split(',')]
    return list(vector_str)


class MomentsService:
    """Service for clustering and generating photo moments."""

    def __init__(self, supabase_client):
        self._client = supabase_client

    async def generate_moments_for_user(
        self,
        user_id: str,
        settings: Dict,
        on_progress=None
    ) -> List[GeneratedMoment]:
        """
        Generate all moments for a user based on their settings.

        Args:
            user_id: The user's ID
            settings: User's moment preferences
            on_progress: Optional callback for progress updates

        Returns:
            List of GeneratedMoment objects
        """
        def report(stage: str, detail: str, progress: int):
            logger.info(f"[moments] {stage}: {detail}")
            if on_progress:
                on_progress(stage, detail, progress)

        moments: List[GeneratedMoment] = []

        # Get user's assets with metadata
        report("Fetching", "Loading user assets...", 5)
        assets = await self._get_user_assets(user_id)
        report("Fetching", f"Found {len(assets)} assets", 10)

        if len(assets) < 4:
            report("Done", "Not enough assets for moments", 100)
            return []

        # Generate location-based moments
        if settings.get('moments_location', True):
            report("Location", "Clustering by location...", 20)
            location_moments = await self._cluster_by_location(user_id, assets)
            moments.extend(location_moments)
            report("Location", f"Found {len(location_moments)} location moments", 35)

        # Generate people-based moments
        if settings.get('moments_people', True):
            report("People", "Grouping by people...", 40)
            people_moments = await self._cluster_by_people(user_id, assets)
            moments.extend(people_moments)
            report("People", f"Found {len(people_moments)} people moments", 55)

        # Generate on-this-day moments
        if settings.get('moments_on_this_day', True):
            report("OnThisDay", "Finding historical moments...", 60)
            otd_moments = await self._find_on_this_day(user_id, assets)
            moments.extend(otd_moments)
            report("OnThisDay", f"Found {len(otd_moments)} on-this-day moments", 75)

        # Generate event-based moments (temporal clustering)
        if settings.get('moments_events', True):
            report("Events", "Detecting events...", 80)
            event_moments = await self._cluster_by_events(user_id, assets)
            moments.extend(event_moments)
            report("Events", f"Found {len(event_moments)} event moments", 95)

        report("Done", f"Generated {len(moments)} total moments", 100)
        return moments

    async def _get_user_assets(self, user_id: str) -> List[Dict]:
        """Get all assets for a user with relevant metadata."""
        all_assets = []
        page_size = 1000
        offset = 0

        while True:
            result = self._client.table("assets") \
                .select("id, path, thumbnail, web_uri, created_at, location_lat, location_lng, location_name, media_type, width, height") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .range(offset, offset + page_size - 1) \
                .execute()

            if not result.data:
                break

            all_assets.extend(result.data)

            if len(result.data) < page_size:
                break
            offset += page_size

        return all_assets

    async def _cluster_by_location(
        self,
        user_id: str,
        assets: List[Dict]
    ) -> List[GeneratedMoment]:
        """
        Cluster assets by geographic location using DBSCAN.

        Groups photos within 5km radius with at least 4 photos.
        """
        moments = []

        # Filter assets with valid GPS coordinates
        geo_assets = [
            a for a in assets
            if a.get('location_lat') and a.get('location_lng')
        ]

        if len(geo_assets) < 4:
            return moments

        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            # Prepare coordinates (lat, lng)
            coords = np.array([
                [a['location_lat'], a['location_lng']]
                for a in geo_assets
            ])

            # DBSCAN with haversine distance approximation
            # 5km radius ≈ 0.045 degrees at equator
            eps_degrees = 0.045
            min_samples = 4

            clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples, metric='euclidean')
            labels = clustering.fit_predict(coords)

            # Group assets by cluster
            clusters: Dict[int, List[Dict]] = {}
            for idx, label in enumerate(labels):
                if label == -1:
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(geo_assets[idx])

            # Create moments from clusters
            for label, cluster_assets in clusters.items():
                if len(cluster_assets) < 4:
                    continue

                # Get representative location name
                location_name = self._get_cluster_location_name(cluster_assets)

                # Calculate date range
                dates = [
                    datetime.fromisoformat(a['created_at'].replace('Z', '+00:00'))
                    for a in cluster_assets if a.get('created_at')
                ]
                date_start = min(dates) if dates else None
                date_end = max(dates) if dates else None

                # Calculate cluster center
                center_lat = np.mean([a['location_lat'] for a in cluster_assets])
                center_lng = np.mean([a['location_lng'] for a in cluster_assets])

                # Select cover photos
                cover_ids = await self._select_cover_photos(
                    user_id,
                    [a['id'] for a in cluster_assets]
                )

                title = self._generate_location_title(location_name)
                subtitle = f"{len(cluster_assets)} photos"
                if location_name and location_name != "Unknown":
                    subtitle = f"{len(cluster_assets)} photos from {location_name}"

                moments.append(GeneratedMoment(
                    grouping_type='location',
                    grouping_criteria={
                        'center_lat': float(center_lat),
                        'center_lng': float(center_lng),
                        'location_name': location_name,
                    },
                    title=title,
                    subtitle=subtitle,
                    cover_asset_ids=cover_ids,
                    all_asset_ids=[a['id'] for a in cluster_assets],
                    date_range_start=date_start,
                    date_range_end=date_end,
                ))

        except ImportError:
            logger.warning("sklearn not available for location clustering")

        return moments

    async def _cluster_by_people(
        self,
        user_id: str,
        assets: List[Dict]
    ) -> List[GeneratedMoment]:
        """
        Group photos by detected people using face_clusters.

        Creates moments for people with 4+ photos.
        """
        moments = []

        # Get face clusters with contact info
        clusters_result = self._client.table("face_clusters") \
            .select("id, contact_id, name, face_count") \
            .eq("user_id", user_id) \
            .gte("face_count", 4) \
            .execute()

        if not clusters_result.data:
            return moments

        for cluster in clusters_result.data:
            cluster_id = cluster['id']
            contact_id = cluster.get('contact_id')
            cluster_name = cluster.get('name')

            # Get face embeddings for this cluster to find asset_ids
            faces_result = self._client.table("face_embeddings") \
                .select("asset_id") \
                .eq("cluster_id", cluster_id) \
                .execute()

            if not faces_result.data:
                continue

            asset_ids = list(set(f['asset_id'] for f in faces_result.data))

            if len(asset_ids) < 4:
                continue

            # Get person name from contact if available
            person_name = cluster_name
            if contact_id and not person_name:
                contact_result = self._client.table("contacts") \
                    .select("display_name") \
                    .eq("id", contact_id) \
                    .single() \
                    .execute()
                if contact_result.data:
                    person_name = contact_result.data.get('display_name')

            if not person_name:
                person_name = "Someone"

            # Get asset dates for range
            assets_result = self._client.table("assets") \
                .select("id, created_at") \
                .in_("id", asset_ids) \
                .execute()

            dates = []
            if assets_result.data:
                dates = [
                    datetime.fromisoformat(a['created_at'].replace('Z', '+00:00'))
                    for a in assets_result.data if a.get('created_at')
                ]

            date_start = min(dates) if dates else None
            date_end = max(dates) if dates else None

            # Select cover photos
            cover_ids = await self._select_cover_photos(user_id, asset_ids)

            title = f"Moments with {person_name}"
            subtitle = f"{len(asset_ids)} photos"

            moments.append(GeneratedMoment(
                grouping_type='people',
                grouping_criteria={
                    'cluster_id': cluster_id,
                    'contact_id': contact_id,
                    'person_name': person_name,
                },
                title=title,
                subtitle=subtitle,
                cover_asset_ids=cover_ids,
                all_asset_ids=asset_ids,
                date_range_start=date_start,
                date_range_end=date_end,
            ))

        return moments

    async def _find_on_this_day(
        self,
        user_id: str,
        assets: List[Dict]
    ) -> List[GeneratedMoment]:
        """
        Find photos from the same day in previous years.

        Creates moments with 3+ photos from the same month/day.
        """
        moments = []
        today = datetime.now()
        current_month = today.month
        current_day = today.day

        # Group assets by year for today's date
        by_year: Dict[int, List[Dict]] = {}

        for asset in assets:
            if not asset.get('created_at'):
                continue

            try:
                dt = datetime.fromisoformat(asset['created_at'].replace('Z', '+00:00'))
                if dt.month == current_month and dt.day == current_day:
                    year = dt.year
                    if year != today.year:
                        if year not in by_year:
                            by_year[year] = []
                        by_year[year].append(asset)
            except (ValueError, TypeError):
                continue

        # Create moments for years with enough photos
        for year, year_assets in by_year.items():
            if len(year_assets) < 3:
                continue

            years_ago = today.year - year

            # Get most common location for subtitle
            locations = [a.get('location_name') for a in year_assets if a.get('location_name')]
            location_text = ""
            if locations:
                from collections import Counter
                most_common = Counter(locations).most_common(1)
                if most_common:
                    location_text = f" in {most_common[0][0]}"

            # Select cover photos
            cover_ids = await self._select_cover_photos(
                user_id,
                [a['id'] for a in year_assets]
            )

            title = f"{years_ago} year{'s' if years_ago > 1 else ''} ago today"
            subtitle = f"{len(year_assets)} photos{location_text}"

            moments.append(GeneratedMoment(
                grouping_type='on_this_day',
                grouping_criteria={
                    'year': year,
                    'month': current_month,
                    'day': current_day,
                    'years_ago': years_ago,
                },
                title=title,
                subtitle=subtitle,
                cover_asset_ids=cover_ids,
                all_asset_ids=[a['id'] for a in year_assets],
                date_range_start=datetime(year, current_month, current_day),
                date_range_end=datetime(year, current_month, current_day, 23, 59, 59),
            ))

        return moments

    async def _cluster_by_events(
        self,
        user_id: str,
        assets: List[Dict]
    ) -> List[GeneratedMoment]:
        """
        Cluster photos into events based on temporal gaps.

        Groups photos with < 4 hour gaps between them.
        """
        moments = []

        # Sort assets by creation date
        dated_assets = [
            a for a in assets if a.get('created_at')
        ]

        if len(dated_assets) < 4:
            return moments

        # Parse and sort by date
        for asset in dated_assets:
            try:
                asset['_dt'] = datetime.fromisoformat(
                    asset['created_at'].replace('Z', '+00:00')
                )
            except (ValueError, TypeError):
                asset['_dt'] = None

        dated_assets = [a for a in dated_assets if a.get('_dt')]
        dated_assets.sort(key=lambda x: x['_dt'])

        # Cluster by 4-hour time gaps
        gap_hours = 4
        max_event_duration_days = 3

        events: List[List[Dict]] = []
        current_event: List[Dict] = []

        for asset in dated_assets:
            if not current_event:
                current_event = [asset]
                continue

            last_dt = current_event[-1]['_dt']
            current_dt = asset['_dt']
            gap = (current_dt - last_dt).total_seconds() / 3600

            # Also check total event duration
            event_duration = (current_dt - current_event[0]['_dt']).days

            if gap <= gap_hours and event_duration <= max_event_duration_days:
                current_event.append(asset)
            else:
                if len(current_event) >= 4:
                    events.append(current_event)
                current_event = [asset]

        # Don't forget the last event
        if len(current_event) >= 4:
            events.append(current_event)

        # Create moments from events (limit to recent ones)
        now = datetime.now(dated_assets[0]['_dt'].tzinfo if dated_assets else None)
        recent_events = [
            e for e in events
            if (now - e[-1]['_dt']).days <= 90  # Last 90 days
        ]

        for event_assets in recent_events[:5]:  # Max 5 event moments
            date_start = event_assets[0]['_dt']
            date_end = event_assets[-1]['_dt']

            # Get location info
            locations = [a.get('location_name') for a in event_assets if a.get('location_name')]
            location_text = ""
            if locations:
                from collections import Counter
                most_common = Counter(locations).most_common(1)
                if most_common:
                    location_text = most_common[0][0]

            # Generate event title
            title = self._generate_event_title(date_start, location_text)
            subtitle = f"{len(event_assets)} photos"
            if location_text:
                subtitle = f"{len(event_assets)} photos at {location_text}"

            # Select cover photos
            cover_ids = await self._select_cover_photos(
                user_id,
                [a['id'] for a in event_assets]
            )

            moments.append(GeneratedMoment(
                grouping_type='event',
                grouping_criteria={
                    'date': date_start.isoformat(),
                    'location': location_text,
                },
                title=title,
                subtitle=subtitle,
                cover_asset_ids=cover_ids,
                all_asset_ids=[a['id'] for a in event_assets],
                date_range_start=date_start.replace(tzinfo=None) if date_start.tzinfo else date_start,
                date_range_end=date_end.replace(tzinfo=None) if date_end.tzinfo else date_end,
            ))

        return moments

    async def _select_cover_photos(
        self,
        user_id: str,
        asset_ids: List[str],
        count: int = 6
    ) -> List[str]:
        """
        Select diverse, high-quality photos for the moment cover.

        Uses CLIP embeddings for diversity if available, otherwise random selection.
        """
        if len(asset_ids) <= count:
            return asset_ids

        try:
            # Try to get CLIP embeddings for diversity selection
            embeddings_result = self._client.table("asset_embeddings") \
                .select("asset_id, embedding") \
                .in_("asset_id", asset_ids) \
                .limit(200) \
                .execute()

            if embeddings_result.data and len(embeddings_result.data) >= count:
                return self._select_diverse_by_embedding(
                    embeddings_result.data, count
                )
        except Exception as e:
            logger.warning(f"Could not use embeddings for cover selection: {e}")

        # Fallback: random selection
        import random
        return random.sample(asset_ids, count)

    def _select_diverse_by_embedding(
        self,
        embedding_data: List[Dict],
        count: int
    ) -> List[str]:
        """Select diverse items using furthest-first traversal on embeddings."""
        if len(embedding_data) <= count:
            return [e['asset_id'] for e in embedding_data]

        # Parse embeddings
        embeddings = []
        asset_ids = []
        for item in embedding_data:
            vec = parse_pgvector(item.get('embedding', ''))
            if vec:
                embeddings.append(vec)
                asset_ids.append(item['asset_id'])

        if len(embeddings) <= count:
            return asset_ids

        embeddings = np.array(embeddings)

        # Normalize for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-10)

        # Furthest-first traversal
        selected_indices = [0]
        remaining = set(range(1, len(embeddings)))

        while len(selected_indices) < count and remaining:
            # Find point furthest from all selected points
            max_min_dist = -1
            best_idx = None

            for idx in remaining:
                # Calculate min distance to any selected point
                min_dist = float('inf')
                for sel_idx in selected_indices:
                    dist = 1 - np.dot(normalized[idx], normalized[sel_idx])
                    min_dist = min(min_dist, dist)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [asset_ids[i] for i in selected_indices]

    def _get_cluster_location_name(self, cluster_assets: List[Dict]) -> str:
        """Get the most common location name from a cluster."""
        locations = [
            a.get('location_name')
            for a in cluster_assets
            if a.get('location_name')
        ]

        if not locations:
            return "Unknown"

        from collections import Counter
        most_common = Counter(locations).most_common(1)
        return most_common[0][0] if most_common else "Unknown"

    def _generate_location_title(self, location_name: str) -> str:
        """Generate a title for a location-based moment."""
        if not location_name or location_name == "Unknown":
            return "A Day Out"

        # Common location title patterns
        prefixes = ["Adventures in", "Memories from", "Time in", "Days in"]
        import random
        prefix = random.choice(prefixes)

        # Simplify location name (take first part before comma)
        simple_name = location_name.split(',')[0].strip()

        return f"{prefix} {simple_name}"

    def _generate_event_title(self, date: datetime, location: str = "") -> str:
        """Generate a title for an event-based moment."""
        # Day of week
        day_name = date.strftime("%A")

        # Time of day
        hour = date.hour
        if hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        # Month and day
        month_day = date.strftime("%B %d")

        if location:
            return f"{day_name} {time_of_day} at {location}"
        else:
            return f"{day_name} {time_of_day}, {month_day}"
