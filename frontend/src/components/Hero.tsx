"use client";

import Image from "next/image";
import type { SearchItem } from "@/types";

interface HeroProps {
  item: SearchItem | null;
}

export default function Hero({ item }: HeroProps) {
  if (!item) {
    return (
      <div className="relative flex h-[70vh] items-end bg-gradient-to-b from-gray-900 to-[#0d0d0d] px-6 pb-16">
        <div className="max-w-xl">
          <h1 className="text-4xl font-bold text-emerald-400 md:text-5xl">
            StreamSage
          </h1>
          <p className="mt-3 text-lg text-gray-400">
            Discover your next favorite movie with AI-powered recommendations.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex h-[70vh] items-end overflow-hidden">
      {/* Background poster */}
      {item.poster_url && (
        <Image
          src={item.poster_url}
          alt={item.title}
          fill
          priority
          sizes="100vw"
          className="object-cover object-top opacity-40"
        />
      )}

      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-[#0d0d0d] via-[#0d0d0d]/60 to-transparent" />
      <div className="absolute inset-0 bg-gradient-to-r from-[#0d0d0d]/80 to-transparent" />

      {/* Content */}
      <div className="relative z-10 max-w-xl px-6 pb-16">
        <h1 className="text-3xl font-bold text-white md:text-5xl">
          {item.title}
        </h1>
        {item.rating && (
          <p className="mt-2 text-sm text-gray-300">
            ⭐ {item.rating.toFixed(1)}
            {item.year && <span className="ml-3">{item.year}</span>}
          </p>
        )}
        <div className="mt-4 flex gap-3">
          <a
            href={`/content/${item.id}`}
            className="rounded bg-white px-6 py-2 text-sm font-semibold text-black transition hover:bg-gray-200"
          >
            ▶ Details
          </a>
          <a
            href={`/content/${item.id}`}
            className="rounded border border-gray-500 bg-gray-800/60 px-6 py-2 text-sm font-semibold text-white transition hover:bg-gray-700/60"
          >
            ℹ More Info
          </a>
        </div>
      </div>
    </div>
  );
}
