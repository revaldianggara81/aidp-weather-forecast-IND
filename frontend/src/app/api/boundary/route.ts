import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export const runtime = "nodejs";

// ---------------------------------------------------------------------------
// Local india_states.geojson lookup
// ---------------------------------------------------------------------------
// The file is read from disk at most once per server process; the parsed
// FeatureCollection and a name-keyed lookup map are cached in module-level
// variables. A cached in-flight promise guards against concurrent
// first-requests racing each other while the file is still being read.
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FeatureLookup = Map<string, any>;

let lookupPromise: Promise<FeatureLookup> | null = null;

function normalize(s: string): string {
  return s
    .normalize("NFD")
    .replace(/\p{Mn}/gu, "")
    .toLowerCase()
    .trim()
    .replace(/\s+/g, " ");
}

async function loadLookup(): Promise<FeatureLookup> {
  if (!lookupPromise) {
    lookupPromise = (async () => {
      const filePath = path.join(process.cwd(), "public", "india_states.geojson");
      const raw = await fs.readFile(filePath, "utf-8");
      const geojson = JSON.parse(raw);

      const map: FeatureLookup = new Map();
      for (const feature of geojson.features) {
        const name = feature?.properties?.name;
        if (typeof name === "string") {
          map.set(normalize(name), feature);
        }
      }
      return map;
    })();
  }
  return lookupPromise;
}

export async function GET(request: NextRequest) {
  const city = request.nextUrl.searchParams.get("city");
  if (!city) return NextResponse.json({ error: "city param required" }, { status: 400 });

  try {
    const lookup = await loadLookup();
    const feature = lookup.get(normalize(city));
    if (!feature) return NextResponse.json({ error: "not found" }, { status: 404 });
    return NextResponse.json(feature.geometry);
  } catch (err) {
    console.error("Boundary lookup failed:", err);
    return NextResponse.json({ error: "fetch failed" }, { status: 500 });
  }
}
