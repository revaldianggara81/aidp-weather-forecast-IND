"use client";

import { useState, useEffect } from "react";
import type { WeatherRow } from "@/lib/types";
import type { CityCoords } from "@/app/api/coords/route";
import HorizonToggle from "@/components/HorizonToggle";
import MapWrapper from "@/components/MapWrapper";

export default function MapPage() {
  const [horizon, setHorizon] = useState<7 | 14 | 30>(30);
  const [data, setData] = useState<WeatherRow[]>([]);
  const [coords, setCoords] = useState<CityCoords>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/api/weather").then((r) => {
        if (!r.ok) throw new Error("Failed to fetch weather data");
        return r.json() as Promise<WeatherRow[]>;
      }),
      fetch("/api/coords").then((r) => {
        if (!r.ok) throw new Error("Failed to fetch city coordinates");
        return r.json() as Promise<CityCoords>;
      }),
    ])
      .then(([weatherData, cityCoords]) => {
        setData(weatherData);
        setCoords(cityCoords);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="h-[calc(100vh-48px)] flex flex-col">
      {/* Controls */}
      <div className="px-6 py-3 bg-white border-b border-slate-200 flex items-center justify-between">
        <span className="text-sm font-medium text-slate-600">
          Anomaly Overview
        </span>
        <HorizonToggle value={horizon} onChange={setHorizon} />
      </div>

      {/* Map */}
      <div className="flex-1">
        {loading && (
          <div className="h-full flex items-center justify-center text-slate-500">
            Loading weather data...
          </div>
        )}
        {error && (
          <div className="h-full flex items-center justify-center text-red-500">
            Error: {error}
          </div>
        )}
        {!loading && !error && (
          <MapWrapper data={data} coords={coords} horizon={horizon} />
        )}
      </div>
    </div>
  );
}
