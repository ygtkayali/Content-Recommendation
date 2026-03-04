"use client";

import { useEffect, useState } from "react";
import Hero from "@/components/Hero";
import ContentRow from "@/components/ContentRow";
import { getTrending, getByGenre, pickRandomGenres } from "@/lib/api";
import type { SearchItem } from "@/types";

interface GenreSection {
  genre: string;
  items: SearchItem[];
}

export default function HomePage() {
  const [trending, setTrending] = useState<SearchItem[]>([]);
  const [genres, setGenres] = useState<GenreSection[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const selectedGenres = pickRandomGenres(4);

        const [trendingData, ...genreResults] = await Promise.all([
          getTrending(20),
          ...selectedGenres.map(async (g) => ({
            genre: g,
            items: await getByGenre(g, 20),
          })),
        ]);

        setTrending(trendingData);
        setGenres(genreResults);
      } catch (err) {
        console.error("Failed to load homepage data:", err);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  const heroItem = trending.length > 0 ? trending[0] : null;

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-emerald-500 border-t-transparent" />
      </div>
    );
  }

  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <Hero item={heroItem} />

      {/* Content Rows */}
      <div className="-mt-16 relative z-10 space-y-2">
        <ContentRow title="Trending Now" items={trending} />

        {genres.map((section) => (
          <ContentRow
            key={section.genre}
            title={section.genre}
            items={section.items}
          />
        ))}
      </div>
    </main>
  );
}
