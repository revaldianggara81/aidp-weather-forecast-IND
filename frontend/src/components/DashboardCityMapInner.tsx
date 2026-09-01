"use client";

import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { GeoJsonObject } from "geojson";
import type { WeatherRow } from "@/lib/types";

interface DashboardCityMapInnerProps {
  lat: number;
  lng: number;
  cityName: string;
  dateRow: WeatherRow | null;
}

// ---------------------------------------------------------------------------
// CSS animation injection
// ---------------------------------------------------------------------------
const ANIM_ID = "wx-anomaly-anim";
const ANIM_CSS = `
@keyframes wx-fall {
  0%   { transform: translateY(-8px); opacity: 0.95; }
  100% { transform: translateY( 8px); opacity: 0.40; }
}
@keyframes wx-pulse {
  0%,100% { transform: scale(1);    opacity: 1;   }
  50%      { transform: scale(1.3); opacity: 0.7; }
}
@keyframes wx-sway {
  0%   { transform: translateX(-7px); opacity: 1; }
  100% { transform: translateX( 7px); opacity: 0.75; }
}
`;

function ensureAnimStyles() {
  if (typeof document === "undefined" || document.getElementById(ANIM_ID)) return;
  const el = document.createElement("style");
  el.id = ANIM_ID;
  el.textContent = ANIM_CSS;
  document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Animated DivIcons
// ---------------------------------------------------------------------------
type AnomalyType = "heat" | "rain" | "wind";

const ICON_CONFIG: Record<AnomalyType, { emoji: string; anim: string; filter?: string }> = {
  heat: { emoji: "🌡️", anim: "wx-pulse 1.4s ease-in-out infinite" },
  rain: { emoji: "🌧️", anim: "wx-fall  1.0s ease-in-out infinite alternate" },
  wind: { emoji: "💨", anim: "wx-sway  0.9s ease-in-out infinite alternate", filter: "brightness(0.3) sepia(1) saturate(3) hue-rotate(170deg)" },
};

function makeAnomalyIcon(type: AnomalyType): L.DivIcon {
  ensureAnimStyles();
  const { emoji, anim, filter } = ICON_CONFIG[type];
  const filterStyle = filter ? `filter:${filter};` : "";
  const html = `<div style="font-size:26px;line-height:1;display:inline-block;animation:${anim};${filterStyle}">${emoji}</div>`;
  return L.divIcon({ html, className: "", iconSize: [32, 32], iconAnchor: [16, 16] });
}

// ---------------------------------------------------------------------------
// Point-in-polygon (ray casting)
// All coordinates stored as [lat, lng] internally.
// ---------------------------------------------------------------------------
type LatLng = [number, number]; // [lat, lng]

function pointInRing(lat: number, lng: number, ring: LatLng[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [iLat, iLng] = ring[i];
    const [jLat, jLng] = ring[j];
    if (
      (iLat > lat) !== (jLat > lat) &&
      lng < ((jLng - iLng) * (lat - iLat)) / (jLat - iLat) + iLng
    ) {
      inside = !inside;
    }
  }
  return inside;
}

interface PolygonRings {
  exterior: LatLng[];
  holes: LatLng[][];
}

function extractPolygons(geojson: object): PolygonRings[] {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g = geojson as any;
  const toLL = (coords: number[][]): LatLng[] => coords.map((c) => [c[1], c[0]]);
  const result: PolygonRings[] = [];

  if (g.type === "Polygon") {
    const [ext, ...holes] = g.coordinates as number[][][];
    result.push({ exterior: toLL(ext), holes: holes.map(toLL) });
  } else if (g.type === "MultiPolygon") {
    for (const poly of g.coordinates as number[][][][]) {
      const [ext, ...holes] = poly;
      result.push({ exterior: toLL(ext), holes: holes.map(toLL) });
    }
  }
  return result;
}

function pointInPolygons(lat: number, lng: number, polys: PolygonRings[]): boolean {
  for (const { exterior, holes } of polys) {
    if (pointInRing(lat, lng, exterior)) {
      if (!holes.some((h) => pointInRing(lat, lng, h))) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Bounding box helper
// ---------------------------------------------------------------------------
interface Bbox {
  minLat: number; maxLat: number;
  minLng: number; maxLng: number;
}

function getBbox(geojson: object): Bbox | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g = geojson as any;
  const pts: LatLng[] = [];

  function flatten(c: unknown): void {
    if (!Array.isArray(c)) return;
    if (typeof c[0] === "number" && typeof c[1] === "number" && c.length >= 2) {
      pts.push([c[1] as number, c[0] as number]);
      return;
    }
    for (const item of c) flatten(item);
  }
  if (g.coordinates) flatten(g.coordinates);
  if (!pts.length) return null;

  return {
    minLat: Math.min(...pts.map((p) => p[0])),
    maxLat: Math.max(...pts.map((p) => p[0])),
    minLng: Math.min(...pts.map((p) => p[1])),
    maxLng: Math.max(...pts.map((p) => p[1])),
  };
}

// ---------------------------------------------------------------------------
// Generate valid positions inside the polygon via grid sampling
// ---------------------------------------------------------------------------
const ICON_TARGET = 9; // max total icons across all anomaly types
const GRID_SIZE  = 13; // 13×13 = 169 candidate points

function getValidPositions(bbox: Bbox, polys: PolygonRings[]): LatLng[] {
  const { minLat, maxLat, minLng, maxLng } = bbox;
  const candidates: LatLng[] = [];

  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const lat = minLat + ((r + 0.5) / GRID_SIZE) * (maxLat - minLat);
      const lng = minLng + ((c + 0.5) / GRID_SIZE) * (maxLng - minLng);
      if (pointInPolygons(lat, lng, polys)) candidates.push([lat, lng]);
    }
  }

  if (candidates.length <= ICON_TARGET) return candidates;

  // Evenly subsample so icons spread across the whole polygon
  const result: LatLng[] = [];
  const step = (candidates.length - 1) / (ICON_TARGET - 1);
  for (let i = 0; i < ICON_TARGET; i++) {
    result.push(candidates[Math.round(i * step)]);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Build final icon list — types assigned round-robin to valid positions
// ---------------------------------------------------------------------------
function buildIconList(
  types: AnomalyType[],
  fallback: LatLng,
  bbox: Bbox | null,
  geojson: object | null,
): { type: AnomalyType; pos: LatLng }[] {
  const n = types.length;
  if (n === 0) return [];

  let positions: LatLng[] = [];

  if (bbox && geojson) {
    const polys = extractPolygons(geojson);
    positions = getValidPositions(bbox, polys);
  }

  // Fallback: simple offset grid from centre if polygon parsing yielded nothing
  if (positions.length === 0 && bbox) {
    const cLat = (bbox.minLat + bbox.maxLat) / 2;
    const cLng = (bbox.minLng + bbox.maxLng) / 2;
    const hLat = (bbox.maxLat - bbox.minLat) / 2;
    const hLng = (bbox.maxLng - bbox.minLng) / 2;
    const offsets: LatLng[] = [
      [-0.55, -0.50], [-0.55,  0.10], [-0.50,  0.60],
      [ 0.05, -0.60], [ 0.00,  0.00], [-0.05,  0.55],
      [ 0.55, -0.55], [ 0.50,  0.05], [ 0.60,  0.50],
    ];
    positions = offsets.map(([dLat, dLng]) => [cLat + dLat * hLat, cLng + dLng * hLng]);
  }

  if (positions.length === 0) return types.map((t) => ({ type: t, pos: fallback }));

  // Assign anomaly types in round-robin order across the spread positions
  return positions.map((pos, i) => ({ type: types[i % n], pos }));
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------
function BoundaryFitter({ geojson, lat, lng }: { geojson: object | null; lat: number; lng: number }) {
  const map = useMap();
  useEffect(() => {
    if (geojson) {
      const layer = L.geoJSON(geojson as Parameters<typeof L.geoJSON>[0]);
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [16, 16] });
        return;
      }
    }
    map.setView([lat, lng], 6);
  }, [geojson, lat, lng, map]);
  return null;
}

function AnomalyMarkers({
  dateRow,
  boundary,
  lat,
  lng,
}: {
  dateRow: WeatherRow | null;
  boundary: object | null;
  lat: number;
  lng: number;
}) {
  if (!dateRow) return null;

  const types: AnomalyType[] = [];
  if (dateRow.ANOMALY_TEMPERATURE) types.push("heat");
  if (dateRow.ANOMALY_PRECIPITATION) types.push("rain");
  if (dateRow.ANOMALY_WIND) types.push("wind");
  if (types.length === 0) return null;

  const bbox = boundary ? getBbox(boundary) : null;
  const items = buildIconList(types, [lat, lng], bbox, boundary);

  return (
    <>
      {items.map(({ type, pos }, i) => (
        <Marker key={`${type}-${i}`} position={pos} icon={makeAnomalyIcon(type)} />
      ))}
    </>
  );
}

// ---------------------------------------------------------------------------
// Boundary fetch
// ---------------------------------------------------------------------------
async function fetchCityBoundary(cityName: string): Promise<object | null> {
  try {
    const res = await fetch(`/api/boundary?city=${encodeURIComponent(cityName)}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function DashboardCityMapInner({ lat, lng, cityName, dateRow }: DashboardCityMapInnerProps) {
  const [boundary, setBoundary] = useState<object | null>(null);

  useEffect(() => {
    setBoundary(null);
    fetchCityBoundary(cityName).then(setBoundary);
  }, [cityName]);

  const hasAnomaly = Boolean(
    dateRow && (dateRow.ANOMALY_TEMPERATURE || dateRow.ANOMALY_PRECIPITATION || dateRow.ANOMALY_WIND),
  );

  const boundaryStyle = hasAnomaly
    ? { color: "#f97316", weight: 3, fillColor: "#f97316", fillOpacity: 0.14, dashArray: "4 3" }
    : { color: "#3b82f6", weight: 2.5, fillColor: "#3b82f6", fillOpacity: 0.08, dashArray: "4 3" };

  return (
    <MapContainer center={[lat, lng]} zoom={9} className="h-full w-full" scrollWheelZoom>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />
      <BoundaryFitter geojson={boundary} lat={lat} lng={lng} />
      {boundary && (
        <GeoJSON
          key={`${cityName}-${hasAnomaly}`}
          data={boundary as GeoJsonObject}
          style={boundaryStyle}
        />
      )}
      <AnomalyMarkers dateRow={dateRow} boundary={boundary} lat={lat} lng={lng} />
    </MapContainer>
  );
}
