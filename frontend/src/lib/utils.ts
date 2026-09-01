import type { WeatherRow } from "./types";

export function filterDataForDashboard(
  data: WeatherRow[],
  city: string,
  horizon: 7 | 14 | 30
): WeatherRow[] {
  const cityData = data.filter((row) => row.CITY === city);

  // Find the boundary between historical and forecast
  const forecastRows = cityData.filter((row) => row.SOURCE === "forecast");
  if (forecastRows.length === 0) return cityData;

  const forecastStart = forecastRows[0].DATE;

  // Show 7 days of history before forecast + N days of forecast
  const forecastStartDate = new Date(forecastStart);
  const histCutoff = new Date(forecastStartDate);
  histCutoff.setDate(histCutoff.getDate() - 7);

  const forecastEnd = new Date(forecastStartDate);
  forecastEnd.setDate(forecastEnd.getDate() + horizon);

  return cityData.filter((row) => {
    const d = new Date(row.DATE);
    return d >= histCutoff && d < forecastEnd;
  });
}

export function getForecastData(
  data: WeatherRow[],
  city: string,
  horizon: 7 | 14 | 30
): WeatherRow[] {
  const cityData = data.filter(
    (row) => row.CITY === city && row.SOURCE === "forecast"
  );
  return cityData.slice(0, horizon);
}

export function hasAnomalies(rows: WeatherRow[]): boolean {
  return rows.some(
    (r) => r.ANOMALY_TEMPERATURE || r.ANOMALY_PRECIPITATION || r.ANOMALY_WIND
  );
}

export function countAnomalies(rows: WeatherRow[]): number {
  return rows.filter(
    (r) => r.ANOMALY_TEMPERATURE || r.ANOMALY_PRECIPITATION || r.ANOMALY_WIND
  ).length;
}

export function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

export function getCitiesFromData(data: WeatherRow[]): string[] {
  return [...new Set(data.map((r) => r.CITY))].sort();
}
