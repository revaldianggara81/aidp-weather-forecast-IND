import { NextRequest, NextResponse } from "next/server";
import { fetchWeatherData } from "@/lib/delta-share";

export async function GET(request: NextRequest) {
  const city = request.nextUrl.searchParams.get("city");

  try {
    const data = await fetchWeatherData();

    if (city) {
      const filtered = data.filter((row) => row.CITY === city);
      return NextResponse.json(filtered);
    }

    return NextResponse.json(data);
  } catch (err) {
    console.error("Failed to fetch weather data:", err);
    return NextResponse.json(
      { error: "Failed to fetch weather data" },
      { status: 500 }
    );
  }
}
