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

  const releaseYear = content.release_date
    ? new Date(content.release_date).getFullYear()
    : null;

  return (
    <main className="min-h-screen pt-20 pb-16">
      {/* ── Detail Section ──────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-6">
        <div className="flex flex-col gap-8 md:flex-row">
          {/* Poster */}
          <div className="relative mx-auto aspect-[2/3] w-64 flex-shrink-0 overflow-hidden rounded-lg bg-gray-800 md:mx-0 md:w-72">
            {content.poster_url ? (
              <Image
                src={content.poster_url}
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

            {/* Meta chips */}
            <div className="flex flex-wrap gap-3 text-sm">
              {releaseYear && (
                <span className="rounded-full bg-gray-800 px-3 py-1 text-gray-300">
                  {releaseYear}
                </span>
              )}
              {content.runtime && (
                <span className="rounded-full bg-gray-800 px-3 py-1 text-gray-300">
                  {Math.floor(content.runtime / 60)}h {content.runtime % 60}m
                </span>
              )}
              {content.language && (
                <span className="rounded-full bg-gray-800 px-3 py-1 text-gray-300 uppercase">
                  {content.language}
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
              {content.vote_average !== null && (
                <div className="flex items-center gap-1">
                  <span className="text-lg text-yellow-400">⭐</span>
                  <span className="text-lg font-semibold text-white">
                    {content.vote_average.toFixed(1)}
                  </span>
                  {content.vote_count !== null && (
                    <span className="ml-1 text-gray-500">
                      ({content.vote_count.toLocaleString()} votes)
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Director & Cast */}
            <div className="flex flex-wrap gap-x-8 gap-y-1 text-sm text-gray-400">
              {content.director && (
                <span>
                  Director:{" "}
                  <span className="text-gray-200">{content.director}</span>
                </span>
              )}
              {content.collection && (
                <span>
                  Collection:{" "}
                  <span className="text-gray-200">{content.collection}</span>
                </span>
              )}
            </div>

            {content.actors.length > 0 && (
              <div className="text-sm text-gray-400">
                <span>Cast: </span>
                <span className="text-gray-200">
                  {content.actors.join(", ")}
                </span>
              </div>
            )}

            {/* Overview */}
            {content.overview && (
              <div className="mt-2">
                <h2 className="mb-1 text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Overview
                </h2>
                <p className="leading-relaxed text-gray-300">
                  {content.overview}
                </p>
              </div>
            )}

            {/* Keywords */}
            {content.keywords.length > 0 && (
              <div className="mt-2">
                <h2 className="mb-2 text-sm font-semibold text-gray-400 uppercase tracking-wide">
                  Keywords
                </h2>
                <div className="flex flex-wrap gap-2">
                  {content.keywords.slice(0, 12).map((k) => (
                    <span
                      key={k}
                      className="rounded bg-gray-800 px-2 py-0.5 text-xs text-gray-400"
                    >
                      {k}
                    </span>
                  ))}
                </div>
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
