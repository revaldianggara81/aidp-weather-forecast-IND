import { NextResponse } from "next/server";
import { fetchWeatherData } from "@/lib/delta-share";
import { getCitiesFromData } from "@/lib/utils";

export type CityCoords = Record<string, { lat: number; lng: number }>;

// Cache geocoded results — invalidated when the process restarts
let cache: { coords: CityCoords; ts: number } | null = null;
const CACHE_TTL = 60 * 60 * 1000; // 1 hour

async function geocodeCity(
  city: string
): Promise<{ lat: number; lng: number } | null> {
  const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const json = await res.json();
    const result = json.results?.[0];
    if (!result) return null;
    return { lat: result.latitude, lng: result.longitude };
  } catch {
    return null;
  }
}

export async function GET() {
  if (cache && Date.now() - cache.ts < CACHE_TTL) {
    return NextResponse.json(cache.coords);
  }

  try {
    const data = await fetchWeatherData();
    const cities = getCitiesFromData(data);

    const entries = await Promise.all(
      cities.map(async (city) => {
        const coords = await geocodeCity(city);
        return [city, coords] as const;
      })
    );

    const coords: CityCoords = Object.fromEntries(
      entries.filter((e): e is [string, { lat: number; lng: number }] => e[1] !== null)
    );

    cache = { coords, ts: Date.now() };
    return NextResponse.json(coords);
  } catch (err) {
    console.error("Failed to geocode cities:", err);
    return NextResponse.json(
      { error: "Failed to fetch city coordinates" },
      { status: 500 }
    );
  }
}
