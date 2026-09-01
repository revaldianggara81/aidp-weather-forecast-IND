"use client";

import Markdown from "react-markdown";
import type { WeatherRow } from "@/lib/types";
import { formatDate } from "@/lib/utils";

interface RecommendationBoxProps {
  selected: WeatherRow | null;
  onClose: () => void;
}

function getAnomalyTypes(row: WeatherRow): string {
  if (row.combined_anomaly) return row.combined_anomaly;
  const types: string[] = [];
  if (row.ANOMALY_TEMPERATURE) types.push("extreme temperature");
  if (row.ANOMALY_PRECIPITATION) types.push("heavy precipitation");
  if (row.ANOMALY_WIND) types.push("strong winds");
  return types.join(", ") || "unknown anomaly";
}

export default function RecommendationBox({
  selected,
  onClose,
}: RecommendationBoxProps) {
  if (!selected) return null;

  const anomalyTypes = getAnomalyTypes(selected);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl shadow-xl max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded-full bg-red-500" />
            <h4 className="text-base font-semibold text-red-800">
              Anomaly Detected — {formatDate(selected.DATE)}
            </h4>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 text-xl leading-none p-1"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        {/* Body */}
        <div className="px-5 py-4">
          <p className="text-sm font-medium text-red-700 capitalize mb-1">
            {anomalyTypes}
          </p>
          <p className="text-xs text-slate-400 mb-4">
            {selected.CITY} &middot;{" "}
            {new Date(selected.DATE).toLocaleDateString("en-US", {
              weekday: "long",
              month: "long",
              day: "numeric",
              year: "numeric",
            })}
          </p>

          {selected.RECOMMENDATION ? (
            <div className="text-sm text-slate-700 leading-relaxed prose prose-sm prose-slate max-w-none">
              <Markdown>{selected.RECOMMENDATION}</Markdown>
            </div>
          ) : (
            <p className="text-sm text-slate-400 italic">
              No detailed recommendation available for this anomaly.
            </p>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-slate-100 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-sm font-medium text-slate-600 bg-slate-100 rounded-md hover:bg-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
