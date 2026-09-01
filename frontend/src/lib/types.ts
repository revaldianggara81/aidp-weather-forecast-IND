export interface WeatherRow {
  DATE: string;
  CITY: string;
  TEMPERATURE_C: number;
  PRECIPITATION_MM: number;
  WIND_SPEED_KMH: number;
  SOURCE: "historical" | "forecast";
  ANOMALY_TEMPERATURE: boolean;
  ANOMALY_PRECIPITATION: boolean;
  ANOMALY_WIND: boolean;
  combined_anomaly: string | null;
  RECOMMENDATION: string | null;
}

export type CityName = string;
