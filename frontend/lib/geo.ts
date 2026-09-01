/** Browser geolocation for SOS SMS (research demo only). */
export type SosGeoPayload = {
  latitude?: number;
  longitude?: number;
  accuracy_m?: number;
  location_label?: string;
  maps_url?: string;
};

export async function captureBrowserLocation(
  timeoutMs = 10000
): Promise<SosGeoPayload | null> {
  if (typeof navigator === "undefined" || !navigator.geolocation) {
    return null;
  }
  try {
    const pos = await new Promise<GeolocationPosition>((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, {
        enableHighAccuracy: true,
        timeout: timeoutMs,
        maximumAge: 30_000,
      });
    });
    const latitude = pos.coords.latitude;
    const longitude = pos.coords.longitude;
    const accuracy_m = pos.coords.accuracy;
    const maps_url = `https://maps.google.com/?q=${latitude.toFixed(5)},${longitude.toFixed(5)}`;
    return {
      latitude,
      longitude,
      accuracy_m,
      maps_url,
      location_label: `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`,
    };
  } catch {
    return null;
  }
}
