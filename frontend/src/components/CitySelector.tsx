"use client";

interface CitySelectorProps {
  cities: string[];
  value: string;
  onChange: (city: string) => void;
}

export default function CitySelector({ cities, value, onChange }: CitySelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <label htmlFor="city-select" className="text-sm font-medium text-slate-600">
        Region:
      </label>
      <select
        id="city-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border border-slate-300 rounded-md px-3 py-1.5 text-sm bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {cities.map((city) => (
          <option key={city} value={city}>
            {city}
          </option>
        ))}
      </select>
    </div>
  );
}
