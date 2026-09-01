"use client";

import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { WeatherRow } from "@/lib/types";
import { formatDate } from "@/lib/utils";

type MetricKey = "TEMPERATURE_C" | "PRECIPITATION_MM" | "WIND_SPEED_KMH";
type AnomalyKey =
  | "ANOMALY_TEMPERATURE"
  | "ANOMALY_PRECIPITATION"
  | "ANOMALY_WIND";

interface WeatherChartProps {
  data: WeatherRow[];
  metric: MetricKey;
  title: string;
  unit: string;
  color: string;
  onAnomalyClick?: (row: WeatherRow) => void;
  onBarClick?: (row: WeatherRow) => void;
  selectedDate?: string | null;
}

const METRIC_TO_ANOMALY: Record<MetricKey, AnomalyKey> = {
  TEMPERATURE_C: "ANOMALY_TEMPERATURE",
  PRECIPITATION_MM: "ANOMALY_PRECIPITATION",
  WIND_SPEED_KMH: "ANOMALY_WIND",
};

interface ChartPoint {
  date: string;
  dateLabel: string;
  value: number;
  historical: number | null;
  forecast: number | null;
  isAnomaly: boolean;
  isForecast: boolean;
  isSelected: boolean;
  row: WeatherRow;
}

export default function WeatherChart({
  data,
  metric,
  title,
  unit,
  color,
  onAnomalyClick,
  onBarClick,
  selectedDate,
}: WeatherChartProps) {
  const anomalyKey = METRIC_TO_ANOMALY[metric];

  // Find the boundary date (last historical date)
  const historicalRows = data.filter((r) => r.SOURCE === "historical");
  const todayDate =
    historicalRows.length > 0
      ? historicalRows[historicalRows.length - 1].DATE
      : null;

  // Build chart data points
  const chartData: ChartPoint[] = data.map((row) => {
    const value = row[metric];
    const isForecast = row.SOURCE === "forecast";
    const isAnomaly = Boolean(row[anomalyKey]) && isForecast;

    return {
      date: row.DATE,
      dateLabel: formatDate(row.DATE),
      value,
      historical: !isForecast ? value : null,
      forecast: isForecast ? value : null,
      isAnomaly,
      isForecast,
      isSelected: isForecast && selectedDate === row.DATE,
      row,
    };
  });

  // Connect trend line across historical/forecast boundary
  if (todayDate) {
    const lastHistIdx = chartData.findIndex((p) => p.date === todayDate);
    if (lastHistIdx >= 0 && lastHistIdx < chartData.length - 1) {
      chartData[lastHistIdx].forecast = chartData[lastHistIdx].historical;
    }
  }

  // Compute Y-axis range — add padding so fluctuations are visible
  const values = chartData.map((p) => p.value);
  const dataMin = Math.min(...values);
  const dataMax = Math.max(...values);
  const padding = (dataMax - dataMin) * 0.1 || 1;
  // For precipitation, keep 0 as the floor
  const yMin = metric === "PRECIPITATION_MM" ? 0 : Math.floor(dataMin - padding);
  const yMax = Math.ceil(dataMax + padding);

  // Lighter color for forecast bars
  const forecastColor = color + "66"; // 40% opacity hex

  return (
    <div className="bg-white rounded-lg border border-slate-200 px-3 py-2 h-full flex flex-col">
      <h3 className="text-xs font-semibold text-slate-700 mb-1">
        {title}{" "}
        <span className="text-slate-400 font-normal">({unit})</span>
      </h3>
      <ResponsiveContainer width="100%" height="100%" minHeight={0} className="flex-1 min-h-0">
        <ComposedChart
          data={chartData}
          margin={{ top: 5, right: 15, bottom: 0, left: -5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="dateLabel"
            tick={{ fontSize: 9, fill: "#94a3b8" }}
            interval="preserveStartEnd"
            tickCount={6}
            height={20}
          />
          <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} width={35} domain={[yMin, yMax]} />
          <Tooltip
            allowEscapeViewBox={{ x: true, y: true }}
            wrapperStyle={{ pointerEvents: "none" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const point = payload[0]?.payload as ChartPoint;
              if (!point) return null;
              return (
                <div className="bg-white border border-slate-200 rounded px-3 py-2 shadow-md text-xs pointer-events-none">
                  <p className="font-medium text-slate-700">
                    {new Date(point.date).toLocaleDateString("en-US", {
                      weekday: "short",
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })}
                  </p>
                  <p className="text-slate-600">
                    {title}: {point.value?.toFixed(1)} {unit}
                  </p>
                  <p className="text-slate-400">
                    {point.isForecast ? "Forecast" : "Historical"}
                  </p>
                  {point.isAnomaly && (
                    <p className="text-red-500 font-medium">
                      Anomaly detected — click bar for details
                    </p>
                  )}
                </div>
              );
            }}
          />
          {todayDate && (
            <ReferenceLine
              x={formatDate(todayDate)}
              stroke="#94a3b8"
              strokeDasharray="4 4"
              label={{
                value: "Today",
                position: "top",
                fill: "#94a3b8",
                fontSize: 9,
              }}
            />
          )}
          {/* Bars — colored by source and anomaly */}
          <Bar
            dataKey="value"
            isAnimationActive={false}
            label={(props: {
              x?: string | number;
              y?: string | number;
              width?: string | number;
              index?: number;
            }) => {
              const x = Number(props.x ?? 0);
              const y = Number(props.y ?? 0);
              const width = Number(props.width ?? 0);
              const index = props.index ?? 0;
              if (!chartData[index]?.isSelected) return <g />;
              const cx = x + width / 2;
              return (
                <polygon
                  points={`${cx - 5},${y - 14} ${cx + 5},${y - 14} ${cx},${y - 6}`}
                  fill="#0f172a"
                />
              );
            }}
            onClick={(barData) => {
              const point = barData as unknown as ChartPoint;
              if (!point) return;
              if (onBarClick) onBarClick(point.row);
              if (point.isAnomaly && onAnomalyClick) onAnomalyClick(point.row);
            }}
          >
            {chartData.map((point, idx) => (
              <Cell
                key={idx}
                fill={
                  point.isAnomaly
                    ? "#ef4444"
                    : point.isForecast
                      ? forecastColor
                      : color
                }
                cursor="pointer"
              />
            ))}
          </Bar>
          {/* Thin dotted trend line — historical */}
          <Line
            type="monotone"
            dataKey="historical"
            stroke={color}
            strokeWidth={1.5}
            strokeDasharray="3 3"
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
          {/* Thin dotted trend line — forecast */}
          <Line
            type="monotone"
            dataKey="forecast"
            stroke={color}
            strokeWidth={1.5}
            strokeDasharray="3 3"
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
