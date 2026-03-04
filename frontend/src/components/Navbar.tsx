"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function Navbar() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    router.push(`/search?q=${encodeURIComponent(trimmed)}`);
    setOpen(false);
  }

  return (
    <nav className="fixed top-0 z-50 flex w-full items-center justify-between bg-gradient-to-b from-black/90 to-transparent px-6 py-4 backdrop-blur-sm">
      {/* Logo */}
      <Link href="/" className="text-2xl font-bold tracking-tight text-emerald-400">
        StreamSage
      </Link>

      {/* Nav links (desktop) */}
      <div className="hidden items-center gap-6 text-sm font-medium text-gray-300 md:flex">
        <Link href="/" className="transition hover:text-white">
          Home
        </Link>
        <Link href="/search?q=" className="transition hover:text-white">
          Browse
        </Link>
      </div>

      {/* Search */}
      <div className="flex items-center gap-3">
        {open ? (
          <form onSubmit={handleSubmit} className="flex items-center">
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Titles, genres..."
              className="w-48 rounded-l border border-gray-600 bg-black/70 px-3 py-1.5 text-sm text-white outline-none placeholder:text-gray-500 focus:border-emerald-500 md:w-64"
            />
            <button
              type="submit"
              className="rounded-r border border-l-0 border-gray-600 bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-emerald-500"
            >
              Search
            </button>
          </form>
        ) : (
          <button
            onClick={() => setOpen(true)}
            aria-label="Open search"
            className="text-gray-300 transition hover:text-white"
          >
            {/* Search icon */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
              className="h-5 w-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
              />
            </svg>
          </button>
        )}
      </div>
    </nav>
  );
}
