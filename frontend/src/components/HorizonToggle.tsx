"use client";

type Horizon = 7 | 14 | 30;

interface HorizonToggleProps {
  value: Horizon;
  onChange: (h: Horizon) => void;
}

const OPTIONS: Horizon[] = [7, 14, 30];

export default function HorizonToggle({ value, onChange }: HorizonToggleProps) {
  return (
    <div className="flex items-center gap-1">
      {OPTIONS.map((h) => (
        <button
          key={h}
          onClick={() => onChange(h)}
          className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
            value === h
              ? "bg-blue-600 text-white"
              : "bg-slate-100 text-slate-600 hover:bg-slate-200"
          }`}
        >
          {h}d
        </button>
      ))}
    </div>
  );
}
