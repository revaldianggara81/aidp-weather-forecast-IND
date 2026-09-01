import type { WeatherRow } from "./types";

const ENDPOINT = process.env.DELTA_SHARE_ENDPOINT!;
const TOKEN = process.env.DELTA_SHARE_TOKEN!;
const SHARE = process.env.DELTA_SHARE_NAME!;
const SCHEMA = process.env.DELTA_SCHEMA_NAME!;
const TABLE = process.env.DELTA_TABLE_NAME!;

// In-memory cache with 1-hour TTL
let cache: { data: WeatherRow[]; ts: number } | null = null;
const CACHE_TTL = 60 * 60 * 1000;

export async function fetchWeatherData(): Promise<WeatherRow[]> {
  if (cache && Date.now() - cache.ts < CACHE_TTL) {
    return cache.data;
  }

  const encodedSchema = encodeURIComponent(SCHEMA);
  const queryUrl = `${ENDPOINT}/shares/${SHARE}/schemas/${encodedSchema}/tables/${TABLE}/query`;

  // Step 1: Query Delta Sharing to get Parquet file URLs
  const res = await fetch(queryUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({}),
  });

  if (!res.ok) {
    throw new Error(
      `Delta Sharing query failed: ${res.status} ${res.statusText}`
    );
  }

  const text = await res.text();
  const lines = text.trim().split("\n");

  // Parse NDJSON — extract column names from metadata and file URLs
  let colNames: string[] = [];
  const fileUrls: string[] = [];

  for (const line of lines) {
    const obj = JSON.parse(line);
    if (obj.metaData?.schemaString) {
      const fields = JSON.parse(obj.metaData.schemaString).fields;
      colNames = fields.map((f: { name: string }) => f.name);
    }
    if (obj.file?.url) {
      fileUrls.push(obj.file.url);
    }
  }

  // Step 2: Download and parse each Parquet file
  const { parquetRead } = await import("hyparquet");
  const { compressors } = await import("hyparquet-compressors");

  const allRows: WeatherRow[] = [];

  for (const url of fileUrls) {
    const parquetRes = await fetch(url);
    if (!parquetRes.ok) {
      throw new Error(`Failed to fetch Parquet file: ${parquetRes.status}`);
    }
    const buffer = await parquetRes.arrayBuffer();

    await parquetRead({
      file: buffer,
      compressors,
      onComplete: (data: unknown[][]) => {
        for (const arr of data) {
          const row: Record<string, unknown> = {};
          colNames.forEach((name, i) => {
            row[name] = arr[i];
          });
          allRows.push({
            DATE:
              row.DATE instanceof Date
                ? row.DATE.toISOString()
                : String(row.DATE),
            CITY: String(row.CITY),
            TEMPERATURE_C: Number(row.TEMPERATURE_C),
            PRECIPITATION_MM: Number(row.PRECIPITATION_MM),
            WIND_SPEED_KMH: Number(row.WIND_SPEED_KMH),
            SOURCE: String(row.SOURCE) as "historical" | "forecast",
            ANOMALY_TEMPERATURE: Boolean(row.ANOMALY_TEMPERATURE),
            ANOMALY_PRECIPITATION: Boolean(row.ANOMALY_PRECIPITATION),
            ANOMALY_WIND: Boolean(row.ANOMALY_WIND),
            combined_anomaly: row.combined_anomaly
              ? String(row.combined_anomaly)
              : null,
            RECOMMENDATION: row.RECOMMENDATION
              ? String(row.RECOMMENDATION)
              : null,
          });
        }
      },
    });
  }

  // Sort by city then date
  allRows.sort((a, b) => {
    if (a.CITY !== b.CITY) return a.CITY.localeCompare(b.CITY);
    return a.DATE.localeCompare(b.DATE);
  });

  cache = { data: allRows, ts: Date.now() };
  return allRows;
}
