"""Natural language date and location parsing utility."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Result of parsing a natural language query."""
    semantic_query: str  # The part for CLIP embedding
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    locations: List[str] = None  # Location terms extracted

    def __post_init__(self):
        if self.locations is None:
            self.locations = []

# Common date patterns
MONTH_NAMES = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}

RELATIVE_TERMS = {
    'today': 0,
    'yesterday': -1,
    'last week': -7,
    'last month': -30,
    'last year': -365,
}


def parse_date_from_query(query: str) -> Tuple[str, Optional[datetime], Optional[datetime]]:
    """
    Extract date range from natural language query.

    Returns:
        Tuple of (cleaned_query, date_start, date_end)
    """
    query_lower = query.lower()
    cleaned_query = query
    date_start = None
    date_end = None

    # Try to match "in [month] [year]" or "[month] [year]"
    month_year_pattern = r'\b(?:in\s+|from\s+)?(' + '|'.join(MONTH_NAMES.keys()) + r')\s+(\d{4})\b'
    match = re.search(month_year_pattern, query_lower)
    if match:
        month_name = match.group(1)
        year = int(match.group(2))
        month = MONTH_NAMES[month_name]

        # Get first and last day of month
        date_start = datetime(year, month, 1)
        if month == 12:
            date_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            date_end = datetime(year, month + 1, 1) - timedelta(days=1)

        # Remove the date part from query
        cleaned_query = re.sub(month_year_pattern, '', query_lower, flags=re.IGNORECASE).strip()
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query)  # Clean up extra spaces
        logger.info(f"Parsed date: {month_name} {year} -> {date_start} to {date_end}")

    # Try to match just year "in 2024" or "from 2024"
    if not date_start:
        year_pattern = r'\b(?:in|from)\s+(\d{4})\b'
        match = re.search(year_pattern, query_lower)
        if match:
            year = int(match.group(1))
            date_start = datetime(year, 1, 1)
            date_end = datetime(year, 12, 31)
            cleaned_query = re.sub(year_pattern, '', query_lower, flags=re.IGNORECASE).strip()
            cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
            logger.info(f"Parsed year: {year} -> {date_start} to {date_end}")

    # Try relative terms
    if not date_start:
        for term, days_offset in RELATIVE_TERMS.items():
            if term in query_lower:
                now = datetime.now()
                if term == 'today':
                    date_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    date_end = now.replace(hour=23, minute=59, second=59)
                elif term == 'yesterday':
                    yesterday = now - timedelta(days=1)
                    date_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
                    date_end = yesterday.replace(hour=23, minute=59, second=59)
                elif term == 'last week':
                    date_end = now
                    date_start = now - timedelta(days=7)
                elif term == 'last month':
                    date_end = now
                    date_start = now - timedelta(days=30)
                elif term == 'last year':
                    date_end = now
                    date_start = now - timedelta(days=365)

                cleaned_query = query_lower.replace(term, '').strip()
                cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
                logger.info(f"Parsed relative date: {term} -> {date_start} to {date_end}")
                break

    # Try "last summer", "last winter", etc.
    if not date_start:
        season_pattern = r'\b(?:last\s+)?(summer|winter|spring|fall|autumn)\s*(?:of\s+)?(\d{4})?\b'
        match = re.search(season_pattern, query_lower)
        if match:
            season = match.group(1)
            year = int(match.group(2)) if match.group(2) else datetime.now().year

            # If "last summer" without year, use previous occurrence
            if 'last' in query_lower and not match.group(2):
                now = datetime.now()
                if season == 'summer' and now.month < 9:
                    year = now.year - 1
                elif season == 'winter' and now.month > 2:
                    year = now.year
                else:
                    year = now.year - 1

            if season == 'summer':
                date_start = datetime(year, 6, 1)
                date_end = datetime(year, 8, 31)
            elif season == 'winter':
                date_start = datetime(year, 12, 1)
                date_end = datetime(year + 1, 2, 28)
            elif season == 'spring':
                date_start = datetime(year, 3, 1)
                date_end = datetime(year, 5, 31)
            elif season in ('fall', 'autumn'):
                date_start = datetime(year, 9, 1)
                date_end = datetime(year, 11, 30)

            cleaned_query = re.sub(season_pattern, '', query_lower, flags=re.IGNORECASE).strip()
            cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
            logger.info(f"Parsed season: {season} {year} -> {date_start} to {date_end}")

    # Try "christmas 2024", "birthday 2023", etc.
    if not date_start:
        holiday_pattern = r'\b(christmas|thanksgiving|halloween|easter|new\s*year)\s*(?:of\s+)?(\d{4})?\b'
        match = re.search(holiday_pattern, query_lower)
        if match:
            holiday = match.group(1).replace(' ', '')
            year = int(match.group(2)) if match.group(2) else datetime.now().year

            if holiday == 'christmas':
                date_start = datetime(year, 12, 20)
                date_end = datetime(year, 12, 31)
            elif holiday == 'thanksgiving':
                # Approximate - 4th Thursday of November
                date_start = datetime(year, 11, 20)
                date_end = datetime(year, 11, 30)
            elif holiday == 'halloween':
                date_start = datetime(year, 10, 25)
                date_end = datetime(year, 10, 31)
            elif holiday == 'newyear':
                date_start = datetime(year, 12, 31)
                date_end = datetime(year + 1, 1, 2)

            if date_start:
                cleaned_query = re.sub(holiday_pattern, '', query_lower, flags=re.IGNORECASE).strip()
                cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
                logger.info(f"Parsed holiday: {holiday} {year} -> {date_start} to {date_end}")

    # Clean up common filler words
    filler_words = ['photos', 'pictures', 'images', 'from', 'in', 'of', 'the', 'with']
    words = cleaned_query.split()
    cleaned_words = [w for w in words if w not in filler_words or len(words) <= 2]
    cleaned_query = ' '.join(cleaned_words) if cleaned_words else query

    return cleaned_query.strip(), date_start, date_end


def parse_location_from_query(query: str) -> Tuple[str, List[str]]:
    """
    Extract location terms from natural language query.

    Handles patterns like:
    - "in Paris"
    - "at the beach"
    - "near Tokyo"
    - "from New York"

    Returns:
        Tuple of (cleaned_query, location_terms)
    """
    query_lower = query.lower()
    cleaned_query = query
    locations = []

    # Words to exclude from location matching (months, time words)
    excluded_words = set(MONTH_NAMES.keys()) | {
        'today', 'yesterday', 'week', 'month', 'year',
        'summer', 'winter', 'spring', 'fall', 'autumn',
        'christmas', 'thanksgiving', 'halloween', 'easter',
    }

    # Pattern: "in/at/near [Location]" where Location is capitalized words
    # Don't match "from" followed by date-like words
    location_pattern = r'\b(?:in|at|near)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
    matches = re.findall(location_pattern, query)
    for match in matches:
        # Skip if it's a month name or time word
        if match.lower() in excluded_words:
            continue
        locations.append(match)
        # Remove from query
        cleaned_query = re.sub(
            r'\b(?:in|at|near)\s+' + re.escape(match) + r'\b',
            '',
            cleaned_query,
            flags=re.IGNORECASE
        )

    # Also try lowercase location pattern for common place types
    place_types = [
        'beach', 'mountain', 'mountains', 'park', 'garden', 'gardens',
        'restaurant', 'cafe', 'hotel', 'airport', 'station',
        'museum', 'church', 'temple', 'castle', 'palace',
        'lake', 'river', 'ocean', 'sea', 'forest', 'woods',
        'city', 'town', 'village', 'downtown', 'suburb',
        'home', 'house', 'apartment', 'office', 'work', 'school',
        'hospital', 'mall', 'store', 'shop', 'market',
        'gym', 'pool', 'stadium', 'arena', 'theater', 'theatre',
        'zoo', 'aquarium', 'farm', 'vineyard', 'winery',
    ]

    place_pattern = r'\b(?:at\s+the\s+|in\s+the\s+|on\s+the\s+|near\s+the\s+)?(' + '|'.join(place_types) + r')s?\b'
    place_matches = re.findall(place_pattern, query_lower)
    for match in place_matches:
        if match not in [loc.lower() for loc in locations]:
            locations.append(match)
            # Don't remove place types from semantic query - they're useful for CLIP

    # Clean up extra spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    if locations:
        logger.info(f"Parsed locations: {locations}")

    return cleaned_query, locations


def parse_query(query: str) -> ParsedQuery:
    """
    Parse a natural language query to extract semantic content, dates, and locations.

    Example:
        "sunset photos in Paris from March 2025"
        -> ParsedQuery(
            semantic_query="sunset",
            date_start=2025-03-01,
            date_end=2025-03-31,
            locations=["Paris"]
        )

    Returns:
        ParsedQuery with all extracted components
    """
    # Extract locations FIRST (before lowercasing) to catch capitalized place names
    cleaned_query, locations = parse_location_from_query(query)

    # Then extract dates from the remaining query
    semantic_query, date_start, date_end = parse_date_from_query(cleaned_query)

    # Clean up the semantic query
    if not semantic_query.strip():
        semantic_query = query  # Fall back to original if nothing left

    return ParsedQuery(
        semantic_query=semantic_query.strip(),
        date_start=date_start,
        date_end=date_end,
        locations=locations
    )
