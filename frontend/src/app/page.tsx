"use client";

import { useState, useEffect } from "react";
import type { WeatherRow } from "@/lib/types";
import type { CityCoords } from "@/app/api/coords/route";
import { filterDataForDashboard, getCitiesFromData, getForecastData } from "@/lib/utils";
import CitySelector from "@/components/CitySelector";
import HorizonToggle from "@/components/HorizonToggle";
import WeatherChart from "@/components/WeatherChart";
import RecommendationBox from "@/components/RecommendationBox";
import DashboardCityMap from "@/components/DashboardCityMap";
import AnomalySummaryBox from "@/components/AnomalySummaryBox";

export default function DashboardPage() {
  const [city, setCity] = useState<string>("");
  const [horizon, setHorizon] = useState<7 | 14 | 30>(30);
  const [allData, setAllData] = useState<WeatherRow[]>([]);
  const [cities, setCities] = useState<string[]>([]);
  const [coords, setCoords] = useState<CityCoords>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAnomaly, setSelectedAnomaly] = useState<WeatherRow | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch("/api/weather").then((res) => {
        if (!res.ok) throw new Error("Failed to fetch data");
        return res.json() as Promise<WeatherRow[]>;
      }),
      fetch("/api/coords").then((res) => {
        if (!res.ok) return {} as CityCoords;
        return res.json() as Promise<CityCoords>;
      }),
    ])
      .then(([data, cityCoords]) => {
        const cityList = getCitiesFromData(data);
        setAllData(data);
        setCities(cityList);
        setCoords(cityCoords);
        setCity((prev) => (prev && cityList.includes(prev) ? prev : cityList[0] ?? ""));
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Reset selectedDate when city changes
  useEffect(() => {
    if (!city || allData.length === 0) return;
    const firstForecast = allData.find((r) => r.CITY === city && r.SOURCE === "forecast");
    setSelectedDate(firstForecast?.DATE ?? null);
  }, [city, allData]);

  const chartData = city ? filterDataForDashboard(allData, city, horizon) : [];
  const forecastRows = city ? getForecastData(allData, city, horizon) : [];
  const cityCoords = city ? (coords[city] ?? null) : null;

  const selectedDateRow = selectedDate
    ? (chartData.find((r) => r.DATE === selectedDate && r.SOURCE === "forecast") ?? null)
    : null;

  const handleBarClick = (row: WeatherRow) => {
    if (row.SOURCE === "forecast") setSelectedDate(row.DATE);
  };

  return (
    <div className="h-[calc(100vh-48px)] flex flex-col bg-slate-50">
      {/* Controls */}
      <div className="px-6 py-2 bg-white border-b border-slate-200 flex items-center justify-between flex-wrap gap-4 shrink-0">
        <CitySelector cities={cities} value={city} onChange={setCity} />
        <HorizonToggle value={horizon} onChange={setHorizon} />
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 px-4 py-3">
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
          <div className="h-full flex gap-3">
            {/* LEFT — 3 charts (60%) */}
            <div className="w-3/5 shrink-0 flex flex-col gap-2 min-h-0">
              <div className="flex-1 min-h-0">
                <WeatherChart
                  data={chartData}
                  metric="TEMPERATURE_C"
                  title="Temperature"
                  unit="°C"
                  color="#f59e0b"
                  onAnomalyClick={setSelectedAnomaly}
                  onBarClick={handleBarClick}
                  selectedDate={selectedDate}
                />
              </div>
              <div className="flex-1 min-h-0">
                <WeatherChart
                  data={chartData}
                  metric="PRECIPITATION_MM"
                  title="Precipitation"
                  unit="mm"
                  color="#3b82f6"
                  onAnomalyClick={setSelectedAnomaly}
                  onBarClick={handleBarClick}
                  selectedDate={selectedDate}
                />
              </div>
              <div className="flex-1 min-h-0">
                <WeatherChart
                  data={chartData}
                  metric="WIND_SPEED_KMH"
                  title="Wind Speed"
                  unit="km/h"
                  color="#10b981"
                  onAnomalyClick={setSelectedAnomaly}
                  onBarClick={handleBarClick}
                  selectedDate={selectedDate}
                />
              </div>
            </div>

            {/* RIGHT — map with anomaly accordion overlay (40%) */}
            <div
              className="flex-1 min-h-0 relative rounded-lg overflow-hidden border border-slate-200"
              style={{ isolation: "isolate" }}
            >
              <DashboardCityMap
                lat={cityCoords?.lat ?? null}
                lng={cityCoords?.lng ?? null}
                cityName={city}
                dateRow={selectedDateRow}
              />
              <AnomalySummaryBox
                forecastRows={forecastRows}
                horizon={horizon}
                selectedDate={selectedDate}
                onDateClick={setSelectedDate}
              />
            </div>
          </div>
        )}
      </div>

      <RecommendationBox
        selected={selectedAnomaly}
        onClose={() => setSelectedAnomaly(null)}
      />
    </div>
  );
}
