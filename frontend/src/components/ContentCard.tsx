"use client";

import Image from "next/image";
import Link from "next/link";
import type { SearchItem, RecommendationItem } from "@/types";

type CardItem = SearchItem | RecommendationItem;

function getPosterUrl(item: CardItem): string | null {
  return item.poster_url ?? null;
}

function getSubtext(item: CardItem): string | null {
  if ("year" in item && item.year) return String(item.year);
  if ("score" in item && typeof item.score === "number")
    return `Score: ${item.score.toFixed(2)}`;
  return null;
}

interface ContentCardProps {
  item: CardItem;
}

export default function ContentCard({ item }: ContentCardProps) {
  const poster = getPosterUrl(item);
  const sub = getSubtext(item);

  return (
    <Link
      href={`/content/${item.id}`}
      className="group flex-shrink-0 w-[150px] md:w-[180px] transition-transform duration-200 hover:scale-105"
    >
      {/* Poster */}
      <div className="relative aspect-[2/3] w-full overflow-hidden rounded-md bg-gray-800">
        {poster ? (
          <Image
            src={poster}
            alt={item.title}
            fill
            sizes="180px"
            className="object-cover transition-opacity group-hover:opacity-80"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-gray-500">
            No Image
          </div>
        )}
      </div>

      {/* Title */}
      <p className="mt-2 truncate text-sm font-medium text-gray-200 group-hover:text-white">
        {item.title}
      </p>

      {/* Subtitle */}
      {sub && (
        <p className="truncate text-xs text-gray-500">{sub}</p>
      )}
    </Link>
  );
}
