"use client";

import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { WeatherRow } from "@/lib/types";
import type { CityCoords } from "@/app/api/coords/route";
import {
  getCitiesFromData,
  getForecastData,
  countAnomalies,
  formatDate,
} from "@/lib/utils";

interface MapInnerProps {
  data: WeatherRow[];
  coords: CityCoords;
  horizon: 7 | 14 | 30;
}

// Pulse parameters scale with how many anomaly days exist
// rings: how many concentric pulse rings
// duration: seconds per pulse cycle (shorter = faster = more urgent)
// animName: which keyframe to use (controls how far the ring expands)
function getPulseConfig(count: number): {
  rings: number;
  duration: number;
  animName: string;
} | null {
  if (count === 0) return null;
  if (count <= 5)  return { rings: 1, duration: 3.0, animName: "lp-sm" };
  if (count <= 15) return { rings: 2, duration: 2.0, animName: "lp-md" };
  return           { rings: 3, duration: 1.2, animName: "lp-lg" };
}

const DOT_SIZE = 22;

function buildMarkerIcon(hasAnomaly: boolean, anomalyCount: number): L.DivIcon {
  const color = hasAnomaly ? "#ef4444" : "#22c55e";
  const pulse = hasAnomaly ? getPulseConfig(anomalyCount) : null;

  let ringsHtml = "";
  if (pulse) {
    for (let i = 0; i < pulse.rings; i++) {
      const delay = ((i * pulse.duration) / pulse.rings).toFixed(2);
      ringsHtml += `
        <div style="
          position:absolute;
          width:${DOT_SIZE}px;height:${DOT_SIZE}px;
          border-radius:50%;
          background:rgba(239,68,68,0.45);
          transform:translate(-50%,-50%);
          animation:${pulse.animName} ${pulse.duration}s ease-out ${delay}s infinite;
        "></div>`;
    }
  }

  const html = `
    <div style="position:relative;width:0;height:0;">
      ${ringsHtml}
      <div style="
        position:absolute;
        width:${DOT_SIZE}px;height:${DOT_SIZE}px;
        border-radius:50%;
        background:${color};
        border:2.5px solid #fff;
        transform:translate(-50%,-50%);
        box-shadow:0 2px 5px rgba(0,0,0,0.35);
      "></div>
    </div>`;

  return L.divIcon({ html, className: "", iconSize: [0, 0], iconAnchor: [0, 0] });
}

// Injected once — three keyframes that expand to different scales
const PULSE_CSS = `
  @keyframes lp-sm {
    0%   { transform:translate(-50%,-50%) scale(1);   opacity:0.7; }
    100% { transform:translate(-50%,-50%) scale(2.8); opacity:0;   }
  }
  @keyframes lp-md {
    0%   { transform:translate(-50%,-50%) scale(1);   opacity:0.75; }
    100% { transform:translate(-50%,-50%) scale(4.0); opacity:0;    }
  }
  @keyframes lp-lg {
    0%   { transform:translate(-50%,-50%) scale(1);   opacity:0.8; }
    100% { transform:translate(-50%,-50%) scale(5.5); opacity:0;   }
  }
`;

export default function MapInner({ data, coords, horizon }: MapInnerProps) {
  // Inject keyframes once
  useEffect(() => {
    if (document.getElementById("map-pulse-css")) return;
    const el = document.createElement("style");
    el.id = "map-pulse-css";
    el.textContent = PULSE_CSS;
    document.head.appendChild(el);
  }, []);

  const cities = getCitiesFromData(data).filter((city) => coords[city]);

  return (
    <MapContainer
      center={[22, 82.5]}
      zoom={4}
      className="h-full w-full"
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {cities.map((city) => {
        const { lat, lng } = coords[city];
        const forecastRows = getForecastData(data, city, horizon);
        const anomalyCount = countAnomalies(forecastRows);
        const hasAnomaly = anomalyCount > 0;

        const anomalyRows = forecastRows.filter(
          (r) =>
            r.ANOMALY_TEMPERATURE ||
            r.ANOMALY_PRECIPITATION ||
            r.ANOMALY_WIND
        );

        return (
          <Marker
            key={city}
            position={[lat, lng]}
            icon={buildMarkerIcon(hasAnomaly, anomalyCount)}
          >
            <Popup>
              <div className="text-sm min-w-[180px]">
                <p className="font-semibold text-base mb-1">{city}</p>
                {hasAnomaly ? (
                  <>
                    <p className="text-red-600 font-medium mb-2">
                      {anomalyCount} anomal{anomalyCount === 1 ? "y" : "ies"} in next{" "}
                      {horizon} days
                    </p>
                    <ul className="space-y-1">
                      {anomalyRows.map((r) => {
                        const types: string[] = [];
                        if (r.ANOMALY_TEMPERATURE) types.push("temp");
                        if (r.ANOMALY_PRECIPITATION) types.push("precip");
                        if (r.ANOMALY_WIND) types.push("wind");
                        return (
                          <li key={r.DATE} className="text-xs text-slate-600">
                            {formatDate(r.DATE)} — {types.join(", ")}
                          </li>
                        );
                      })}
                    </ul>
                  </>
                ) : (
                  <p className="text-green-600 font-medium">
                    No anomalies in next {horizon} days
                  </p>
                )}
              </div>
            </Popup>
          </Marker>
        );
      })}
    </MapContainer>
  );
}
