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
  english_name: string | null;
  score: number | null;
  genres: string[];
  synopsis: string | null;
  type: string | null;
  studios: string | null;
  source: string | null;
  popularity: number | null;
  scored_by: number | null;
  image_url: string | null;
  aired: string | null;
  premiered: string | null;
  status: string | null;
  rank: number | null;
}

export interface RecommendationItem {
  id: number;
  title: string;
  score: number;
  poster_url: string | null;
}
