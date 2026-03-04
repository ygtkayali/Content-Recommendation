"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { useParams } from "next/navigation";
import ContentRow from "@/components/ContentRow";
import { getContent, getRecommendations } from "@/lib/api";
import type { ContentDetail, RecommendationItem } from "@/types";

export default function ContentPage() {
  const params = useParams();
  const id = Number(params.id);

  const [content, setContent] = useState<ContentDetail | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>(
    [],
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isNaN(id)) {
      setError("Invalid content ID.");
      setLoading(false);
      return;
    }

    async function load() {
      try {
        const [detail, recs] = await Promise.all([
          getContent(id),
          getRecommendations(id, 20),
        ]);
        setContent(detail);
        setRecommendations(recs);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load content.");
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [id]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center pt-20">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-emerald-500 border-t-transparent" />
      </div>
    );
  }

  if (error || !content) {
    return (
      <div className="flex min-h-screen items-center justify-center pt-20">
        <p className="text-lg text-red-400">{error ?? "Content not found."}</p>
      </div>
    );
  }

  return (
    <main className="min-h-screen pt-20 pb-16">
      {/* ── Detail Section ──────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-6">
        <div className="flex flex-col gap-8 md:flex-row">
          {/* Poster */}
          <div className="relative mx-auto aspect-[2/3] w-64 flex-shrink-0 overflow-hidden rounded-lg bg-gray-800 md:mx-0 md:w-72">
            {content.image_url ? (
              <Image
                src={content.image_url}
                alt={content.title}
                fill
                priority
                sizes="288px"
                className="object-cover"
              />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-gray-500">
                No Image
              </div>
            )}
          </div>

          {/* Info */}
          <div className="flex flex-1 flex-col gap-4">
            {/* Title */}
            <h1 className="text-3xl font-bold text-white md:text-4xl">
              {content.title}
            </h1>

            {content.english_name && content.english_name !== content.title && (
              <p className="text-lg text-gray-400">{content.english_name}</p>
            )}

            {/* Meta chips */}
            <div className="flex flex-wrap gap-3 text-sm">
              {content.type && (
                <span className="rounded-full bg-gray-800 px-3 py-1 text-gray-300">
                  {content.type}
                </span>
              )}
              {content.status && (
                <span className="rounded-full bg-gray-800 px-3 py-1 text-gray-300">
                  {content.status}
                </span>
              )}
              {content.aired && (
                <span className="rounded-full bg-gray-800 px-3 py-1 text-gray-300">
                  📅 {content.aired}
                </span>
              )}
              {content.premiered && (
                <span className="rounded-full bg-gray-800 px-3 py-1 text-gray-300">
                  {content.premiered}
                </span>
              )}
            </div>

            {/* Genres */}
            {content.genres.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {content.genres.map((g) => (
                  <span
                    key={g}
                    className="rounded bg-emerald-900/50 px-2.5 py-1 text-xs font-medium text-emerald-300"
                  >
                    {g}
                  </span>
                ))}
              </div>
            )}

            {/* Stats row */}
            <div className="flex flex-wrap items-center gap-6 text-sm text-gray-300">
              {content.score !== null && (
                <div className="flex items-center gap-1">
                  <span className="text-lg text-yellow-400">⭐</span>
                  <span className="text-lg font-semibold text-white">
                    {content.score.toFixed(1)}
                  </span>
                  {content.scored_by !== null && (
                    <span className="ml-1 text-gray-500">
                      ({content.scored_by.toLocaleString()} votes)
                    </span>
                  )}
                </div>
              )}
              {content.rank !== null && (
                <span>
                  Rank: <strong className="text-white">#{content.rank}</strong>
                </span>
              )}
              {content.popularity !== null && (
                <span>
                  Popularity:{" "}
                  <strong className="text-white">#{content.popularity}</strong>
                </span>
              )}
            </div>

            {/* Extra info */}
            <div className="flex flex-wrap gap-x-8 gap-y-1 text-sm text-gray-400">
              {content.studios && (
                <span>
                  Studios: <span className="text-gray-200">{content.studios}</span>
                </span>
              )}
              {content.source && (
                <span>
                  Source: <span className="text-gray-200">{content.source}</span>
                </span>
              )}
            </div>

            {/* Synopsis */}
            {content.synopsis && (
              <div className="mt-2">
                <h2 className="mb-1 text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Synopsis
                </h2>
                <p className="leading-relaxed text-gray-300">
                  {content.synopsis}
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ── Similar Content ─────────────────────────────────── */}
      {recommendations.length > 0 && (
        <section className="mt-12">
          <ContentRow title="You Might Also Like" items={recommendations} />
        </section>
      )}
    </main>
  );
}
