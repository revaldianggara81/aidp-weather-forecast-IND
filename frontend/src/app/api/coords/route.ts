import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { fetchWeatherData } from "@/lib/delta-share";
import { getCitiesFromData } from "@/lib/utils";

export const runtime = "nodejs";

export type CityCoords = Record<string, { lat: number; lng: number }>;

// Cache coordinate results — invalidated when the process restarts
let cache: { coords: CityCoords; ts: number } | null = null;
const CACHE_TTL = 60 * 60 * 1000; // 1 hour

// ---------------------------------------------------------------------------
// Local india_region_coords.json lookup
// ---------------------------------------------------------------------------
// The file is read from disk at most once per server process; the parsed
// name-keyed lookup map is cached in a module-level variable. A cached
// in-flight promise guards against concurrent first-requests racing each
// other while the file is still being read.
// ---------------------------------------------------------------------------

type CoordLookup = Map<string, { lat: number; lng: number }>;

let lookupPromise: Promise<CoordLookup> | null = null;

function normalize(s: string): string {
  return s
    .normalize("NFD")
    .replace(/\p{Mn}/gu, "")
    .toLowerCase()
    .trim()
    .replace(/\s+/g, " ");
}

async function loadLookup(): Promise<CoordLookup> {
  if (!lookupPromise) {
    lookupPromise = (async () => {
      const filePath = path.join(process.cwd(), "public", "india_region_coords.json");
      const raw = await fs.readFile(filePath, "utf-8");
      const json = JSON.parse(raw) as Record<string, { lat: number; lng: number }>;

      const map: CoordLookup = new Map();
      for (const [name, coords] of Object.entries(json)) {
        map.set(normalize(name), coords);
      }
      return map;
    })();
  }
  return lookupPromise;
}

export async function GET() {
  if (cache && Date.now() - cache.ts < CACHE_TTL) {
    return NextResponse.json(cache.coords);
  }

  try {
    const data = await fetchWeatherData();
    const cities = getCitiesFromData(data);
    const lookup = await loadLookup();

    const entries = cities
      .map((city) => {
        const coords = lookup.get(normalize(city));
        if (!coords) {
          console.warn(`No coordinates found for region: ${city}`);
          return null;
        }
        return [city, coords] as const;
      })
      .filter((e): e is [string, { lat: number; lng: number }] => e !== null);

    const coords: CityCoords = Object.fromEntries(entries);

    cache = { coords, ts: Date.now() };
    return NextResponse.json(coords);
  } catch (err) {
    console.error("Failed to look up city coordinates:", err);
    return NextResponse.json(
      { error: "Failed to fetch city coordinates" },
      { status: 500 }
    );
  }
}
