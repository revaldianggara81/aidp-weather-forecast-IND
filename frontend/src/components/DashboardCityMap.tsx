"use client";

import dynamic from "next/dynamic";
import type { WeatherRow } from "@/lib/types";

const DashboardCityMapInner = dynamic(
  () => import("./DashboardCityMapInner"),
  {
    ssr: false,
    loading: () => (
      <div className="h-full w-full flex items-center justify-center bg-slate-100 text-slate-400 text-sm">
        Loading map...
      </div>
    ),
  }
);

interface DashboardCityMapProps {
  lat: number | null;
  lng: number | null;
  cityName: string;
  dateRow: WeatherRow | null;
}

export default function DashboardCityMap({
  lat,
  lng,
  cityName,
  dateRow,
}: DashboardCityMapProps) {
  if (lat === null || lng === null) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-slate-100 text-slate-400 text-sm">
        Locating city...
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      <DashboardCityMapInner
        lat={lat}
        lng={lng}
        cityName={cityName}
        dateRow={dateRow}
      />
    </div>
  );
}
