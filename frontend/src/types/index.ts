// Types matching the FastAPI backend response models

export interface SearchItem {
  id: number;
  title: string;
  year: number | null;
  rating: number | null;
  poster_url: string | null;
}

export interface ContentDetail {
  id: number;
  title: string;
  overview: string | null;
  genres: string[];
  keywords: string[];
  actors: string[];
  director: string | null;
  collection: string | null;
  vote_average: number | null;
  vote_count: number | null;
  runtime: number | null;
  release_date: string | null;
  language: string | null;
  poster_url: string | null;
}

export interface RecommendationItem {
  id: number;
  title: string;
  score: number;
  poster_url: string | null;
}
