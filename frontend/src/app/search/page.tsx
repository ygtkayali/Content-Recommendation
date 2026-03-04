"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import ContentCard from "@/components/ContentCard";
import { searchContent } from "@/lib/api";
import type { SearchItem } from "@/types";

function SearchResults() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const q = searchParams.get("q") ?? "";

  const [query, setQuery] = useState(q);
  const [results, setResults] = useState<SearchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  useEffect(() => {
    setQuery(q);
    if (q.trim()) {
      setLoading(true);
      searchContent(q, 40)
        .then(setResults)
        .catch(() => setResults([]))
        .finally(() => {
          setLoading(false);
          setSearched(true);
        });
    }
  }, [q]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    router.push(`/search?q=${encodeURIComponent(trimmed)}`);
  }

  return (
    <main className="min-h-screen px-6 pt-24 pb-16">
      {/* Search input */}
      <form onSubmit={handleSubmit} className="mx-auto mb-10 max-w-xl">
        <div className="flex">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search anime titles..."
            className="flex-1 rounded-l-lg border border-gray-700 bg-gray-900 px-4 py-3 text-white outline-none placeholder:text-gray-500 focus:border-emerald-500"
          />
          <button
            type="submit"
            className="rounded-r-lg bg-emerald-600 px-6 py-3 font-medium text-white transition hover:bg-emerald-500"
          >
            Search
          </button>
        </div>
      </form>

      {/* Results */}
      {loading ? (
        <div className="flex justify-center">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-emerald-500 border-t-transparent" />
        </div>
      ) : results.length > 0 ? (
        <>
          <p className="mb-6 text-sm text-gray-400">
            {results.length} result{results.length !== 1 ? "s" : ""} for &quot;{q}&quot;
          </p>
          <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
            {results.map((item) => (
              <ContentCard key={item.id} item={item} />
            ))}
          </div>
        </>
      ) : searched ? (
        <p className="text-center text-gray-500">
          No results found for &quot;{q}&quot;.
        </p>
      ) : null}
    </main>
  );
}

export default function SearchPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-emerald-500 border-t-transparent" />
        </div>
      }
    >
      <SearchResults />
    </Suspense>
  );
}
