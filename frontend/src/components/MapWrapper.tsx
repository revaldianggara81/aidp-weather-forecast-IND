"use client";

import dynamic from "next/dynamic";
import type { WeatherRow } from "@/lib/types";
import type { CityCoords } from "@/app/api/coords/route";

const MapInner = dynamic(() => import("./MapInner"), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center bg-slate-100 text-slate-400">
      Loading map...
    </div>
  ),
});

interface MapWrapperProps {
  data: WeatherRow[];
  coords: CityCoords;
  horizon: 7 | 14 | 30;
}

export default function MapWrapper({ data, coords, horizon }: MapWrapperProps) {
  return <MapInner data={data} coords={coords} horizon={horizon} />;
}
