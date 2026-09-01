"use client";

import { useState, useRef } from "react";
import type { WeatherRow } from "@/lib/types";
import { formatDate } from "@/lib/utils";

interface AnomalySummaryBoxProps {
  forecastRows: WeatherRow[];
  horizon: 7 | 14 | 30;
  selectedDate: string | null;
  onDateClick: (date: string) => void;
}

function formatDateWithYear(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function getAnomalyIcons(row: WeatherRow): string {
  const icons: string[] = [];
  if (row.ANOMALY_TEMPERATURE) icons.push("🌡️");
  if (row.ANOMALY_PRECIPITATION) icons.push("🌧️");
  if (row.ANOMALY_WIND) icons.push("💨");
  return icons.join("");
}

export default function AnomalySummaryBox({
  forecastRows,
  horizon,
  selectedDate,
  onDateClick,
}: AnomalySummaryBoxProps) {
  const [isOpen, setIsOpen] = useState(true);
  const ref = useRef<HTMLDivElement>(null);

  const anomalyRows = forecastRows
    .slice(0, horizon)
    .filter((r) => r.ANOMALY_TEMPERATURE || r.ANOMALY_PRECIPITATION || r.ANOMALY_WIND);

  const selectedLabel = selectedDate ? formatDateWithYear(selectedDate) : null;
  const hasAnomalies = anomalyRows.length > 0;

  return (
    <div ref={ref} className="absolute top-0 left-0 right-0 rounded-t-lg shadow-md" style={{ zIndex: 1000 }}>
      {/* Header */}
      <button
        onClick={() => setIsOpen((o) => !o)}
        className={`w-full px-3 py-2 flex items-center justify-between transition-colors ${
          hasAnomalies
            ? "bg-red-50 hover:bg-red-100 border-b border-red-200"
            : "bg-green-50 hover:bg-green-100 border-b border-green-200"
        }`}
      >
        <span className={`text-xs font-bold tracking-wide uppercase ${hasAnomalies ? "text-red-700" : "text-green-700"}`}>
          {hasAnomalies
            ? `⚠️ ${anomalyRows.length} Anomal${anomalyRows.length === 1 ? "y" : "ies"} · Next ${horizon} Days`
            : `✅ No Anomalies · Next ${horizon} Days`}
        </span>
        <div className="flex items-center gap-2 shrink-0 ml-2">
          {selectedLabel && (
            <span className="text-xs font-medium text-slate-500 bg-white/80 border border-slate-200 rounded px-1.5 py-0.5">
              {selectedLabel}
            </span>
          )}
          <svg
            className={`w-3.5 h-3.5 transition-transform duration-200 ${isOpen ? "rotate-180" : ""} ${hasAnomalies ? "text-red-400" : "text-green-400"}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Body — pill grid */}
      {isOpen && (
        <div className="bg-white border-b border-slate-200 px-2 py-2">
          {anomalyRows.length === 0 ? (
            <p className="text-xs text-green-600 px-1">✅ No anomalies detected in this period</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {anomalyRows.map((row) => {
                const isSelected = row.DATE === selectedDate;
                return (
                  <button
                    key={row.DATE}
                    onClick={() => onDateClick(row.DATE)}
                    className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border transition-colors ${
                      isSelected
                        ? "bg-violet-100 border-violet-500 text-violet-700"
                        : "bg-red-50 border-red-200 text-red-700 hover:bg-red-100"
                    }`}
                  >
                    {formatDateWithYear(row.DATE)}
                    <span>{getAnomalyIcons(row)}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
