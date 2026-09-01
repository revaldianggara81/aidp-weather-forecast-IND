export default function ArchitecturePage() {
  const steps = [
    {
      label: "Open-Meteo API",
      subtitle: "External Data Source",
      color: "bg-sky-600",
    },
    {
      label: "Bronze",
      subtitle: "Raw Ingestion",
      color: "bg-amber-700",
    },
    {
      label: "Silver",
      subtitle: "Cleaned Data",
      color: "bg-slate-400",
    },
    {
      label: "Gold",
      subtitle: "ML Forecasts",
      color: "bg-yellow-500",
    },
    {
      label: "Delta Sharing",
      subtitle: "Data Delivery",
      color: "bg-emerald-600",
    },
    {
      label: "Frontend",
      subtitle: "Dashboard & Map",
      color: "bg-indigo-600",
    },
  ];

  return (
    <main className="min-h-screen bg-slate-950 text-white px-6 py-10 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">System Architecture</h1>
      <p className="text-slate-400 mb-10">
        End-to-end pipeline from weather data ingestion through the Medallion
        architecture to the interactive frontend.
      </p>

      {/* Flow Diagram */}
      <div className="flex items-center justify-between gap-0 overflow-x-auto pb-4 mb-12">
        {steps.map((step, i) => (
          <div key={step.label} className="flex items-center shrink-0">
            <div
              className={`${step.color} rounded-lg px-5 py-4 text-center min-w-[130px]`}
            >
              <div className="font-semibold text-sm">{step.label}</div>
              <div className="text-xs text-white/70 mt-0.5">
                {step.subtitle}
              </div>
            </div>
            {i < steps.length - 1 && (
              <svg
                className="w-8 h-8 text-slate-500 shrink-0 mx-1"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 5l7 7-7 7"
                />
              </svg>
            )}
          </div>
        ))}
      </div>

      {/* Detailed Sections */}
      <div className="space-y-10">
        <Section
          color="bg-amber-700"
          title="Bronze Layer — Data Ingestion"
          items={[
            "Source: Open-Meteo Archive API (5 years of daily historical weather, batched by coordinate)",
            "Covers all 36 Indian states and union territories, each represented by its administrative capital",
            "Ingests raw weather variables: temperature (°C), precipitation (mm), wind speed (km/h)",
            "Stores data as-is in the Bronze table — no transformations applied",
          ]}
        />

        <Section
          color="bg-slate-400"
          title="Silver Layer — Data Cleaning"
          items={[
            "Cleans raw Bronze data: handles missing values and removes duplicate records",
            "No normalization or feature engineering applied at this stage",
            "Produces a clean, analysis-ready dataset for downstream ML models",
          ]}
        />

        <Section
          color="bg-yellow-500"
          title="Gold Layer — ML Forecasting & Anomaly Detection"
          items={[
            "Trains a multivariate AR-LSTM model on full historical data for each region",
            "Produces 30-day forecasts with Gaussian noise for temperature, precipitation, and wind speed",
            "Anomaly detection using 95th-percentile thresholds on forecasted values",
            "LLM-generated safety recommendations for anomaly days via query_model (OCI GenAI)",
            "Publishes the final forecast + anomaly table to Delta Sharing",
          ]}
        />

        <Section
          color="bg-emerald-600"
          title="Data Delivery — Delta Sharing"
          items={[
            "Gold layer result table is added to AIDP's Data Sharing feature for external access",
            "Data is shared externally via the Delta Sharing protocol",
            "Frontend API route fetches Parquet files from the share endpoint, parsed and cached server-side with a 1-hour TTL",
          ]}
        />

        <Section
          color="bg-indigo-600"
          title="Frontend — Dashboard & Map"
          items={[
            "Next.js application consuming Delta Sharing data via internal API route",
            "Dashboard: bar charts with trend lines, anomaly day highlights, and recommendation dialogs",
            "Map: interactive Leaflet map with region markers colored by anomaly status",
            "Region selector and horizon toggle (7 / 14 / 30 days) for filtering forecasts",
          ]}
        />
      </div>
    </main>
  );
}

function Section({
  color,
  title,
  items,
}: {
  color: string;
  title: string;
  items: string[];
}) {
  return (
    <div className="border border-slate-700 rounded-lg p-6">
      <div className="flex items-center gap-3 mb-4">
        <span className={`${color} w-3 h-3 rounded-full shrink-0`} />
        <h2 className="text-xl font-semibold">{title}</h2>
      </div>
      <ul className="space-y-2 text-slate-300 text-sm">
        {items.map((item) => (
          <li key={item} className="flex gap-2">
            <span className="text-slate-500 mt-0.5">—</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
