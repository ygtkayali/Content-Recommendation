"use client";

import ContentCard from "./ContentCard";
import type { SearchItem, RecommendationItem } from "@/types";

interface ContentRowProps {
  title: string;
  items: (SearchItem | RecommendationItem)[];
}

export default function ContentRow({ title, items }: ContentRowProps) {
  if (items.length === 0) return null;

  return (
    <section className="mb-8">
      <h2 className="mb-3 px-6 text-lg font-semibold text-white md:text-xl">
        {title}
      </h2>

      <div className="flex gap-4 overflow-x-auto px-6 pb-4 scrollbar-thin scrollbar-thumb-gray-700">
        {items.map((item) => (
          <ContentCard key={item.id} item={item} />
        ))}
      </div>
    </section>
  );
}
