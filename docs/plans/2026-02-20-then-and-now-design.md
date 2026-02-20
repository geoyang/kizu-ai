# Then and Now Memory — Design

## Concept

A date-triggered memory that pairs an old photo of a person (3+ years ago today) with a recent photo of the same person. The card shows a side-by-side pair. Tapping in reveals a timeline collage of that person across all years.

## Approach

Date-triggered face lookup (builds on existing On This Day logic). During moment generation, find photos from this calendar date in past years that contain a detected face. Look up the person's cluster, find a recent photo of them, and emit the pair.

## Data Flow

```
generate_moments_for_user()
  -> _find_then_and_now()
  -> Query face_embeddings joined with assets
     WHERE month/day = today AND created_at <= now() - 3 years
  -> For each face: get cluster_id / contact_id
  -> Find recent photo (last 12 months) of same person
  -> Prefer solo/prominent faces, fall back to group
  -> Emit GeneratedMoment(grouping_type='then_and_now')
```

## Photo Selection

### "Then" photo (old)
1. Query `face_embeddings` + `assets` where month/day matches today, created_at <= 3 years ago
2. Rank by face bounding box area (larger = more prominent, prefer solo/close-up)
3. Pick best one per person (cluster_id)

### "Now" photo (recent)
1. Query `face_embeddings` for same cluster_id where created_at >= 12 months ago
2. Rank by bounding box area (prefer prominent)
3. Fall back to any photo with their face if no solo shots
4. Pick most recent prominent one

### Timeline collage (expanded view)
1. Query all `face_embeddings` for that cluster_id, joined with assets
2. Group by year
3. Select 1 best photo per year (largest face bounding box)
4. Return as `timeline_asset_ids` ordered chronologically

## GeneratedMoment Structure

```python
grouping_type: 'then_and_now'
grouping_criteria: {
    "cluster_id": str,
    "contact_id": str | None,
    "person_name": str,
    "years_ago": int,
    "then_asset_id": str,
    "now_asset_id": str,
    "timeline_asset_ids": [str, ...]  # 1 per year, chronological
}
title: "Then and Now"
subtitle: "{person_name} - {years_ago} years apart"
cover_asset_ids: [then_asset_id, now_asset_id]
all_asset_ids: [then_asset_id, now_asset_id, *timeline_asset_ids]
date_range_start: then_photo.created_at
date_range_end: now_photo.created_at
```

## Code Location

New method `_find_then_and_now()` in `api/services/moments_service.py`, called from `generate_moments_for_user()` alongside the existing 4 memory types. Follows the same pattern as `_find_on_this_day()`.

## Deduplication

- One Then and Now per person per day
- Skip people already featured in a people-based moment in the same run
- If same person qualifies for both On This Day and Then and Now, prefer Then and Now

## Edge Cases

- **No recent photos of person**: Skip
- **Person not clustered**: Skip (requires face_clusters data)
- **Multiple people qualify**: Generate one per person, cap at 3 per run
- **Cluster has no name**: Prefer clusters linked to a contact; use "Someone" as fallback or skip
