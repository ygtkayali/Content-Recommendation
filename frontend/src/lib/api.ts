import type { SearchItem, ContentDetail, RecommendationItem } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }

  return res.json() as Promise<T>;
}

// ── Search ────────────────────────────────────────────────
export function searchContent(
  query: string,
  limit = 20,
): Promise<SearchItem[]> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return fetchJSON<SearchItem[]>(`/api/v1/search?${params}`);
}

// ── Content Detail ────────────────────────────────────────
export function getContent(id: number): Promise<ContentDetail> {
  return fetchJSON<ContentDetail>(`/api/v1/content/${id}`);
}

// ── Recommendations ───────────────────────────────────────
export function getRecommendations(
  id: number,
  limit = 10,
): Promise<RecommendationItem[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  return fetchJSON<RecommendationItem[]>(`/api/v1/recommend/${id}?${params}`);
}

// ── Genre Browse ──────────────────────────────────────────
export function getByGenre(
  genre: string,
  limit = 20,
  sortBy: "bayesian" | "popularity" = "bayesian",
): Promise<SearchItem[]> {
  const params = new URLSearchParams({
    limit: String(limit),
    sort_by: sortBy,
  });
  return fetchJSON<SearchItem[]>(
    `/api/v1/genre/${encodeURIComponent(genre)}?${params}`,
  );
}

// ── Trending ──────────────────────────────────────────────
export function getTrending(
  limit = 20,
  sortBy: "bayesian" | "popularity" = "popularity",
): Promise<SearchItem[]> {
  const params = new URLSearchParams({
    limit: String(limit),
    sort_by: sortBy,
  });
  return fetchJSON<SearchItem[]>(`/api/v1/trending?${params}`);
}

// ── Available Genres (client-side constant) ───────────────
export const ALL_GENRES = [
  "Action",
  "Adventure",
  "Animation",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Family",
  "Fantasy",
  "Foreign",
  "History",
  "Horror",
  "Music",
  "Mystery",
  "Romance",
  "Science Fiction",
  "Thriller",
  "TV Movie",
  "War",
  "Western",
] as const;

/** Pick `count` random genres from the full list. */
export function pickRandomGenres(count = 4): string[] {
  const shuffled = [...ALL_GENRES].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}
