"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  const links = [
    { href: "/", label: "Dashboard" },
    { href: "/map", label: "Region Map" },
    { href: "/architecture", label: "Architecture" },
  ];

  return (
    <nav className="bg-slate-800 text-white px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-6">
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={`text-sm font-medium px-3 py-1.5 rounded transition-colors ${
              pathname === link.href
                ? "bg-slate-600 text-white"
                : "text-slate-300 hover:text-white hover:bg-slate-700"
            }`}
          >
            {link.label}
          </Link>
        ))}
      </div>
      <span className="text-sm font-semibold text-slate-300">
        Oracle AICEC AIDP Demo: India Weather Forecast
      </span>
    </nav>
  );
}
