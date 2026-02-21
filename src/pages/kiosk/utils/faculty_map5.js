import L from "leaflet";

let map;

function initMap(mapDivId) {
  const southWest = L.latLng(7.252000, 80.590528);
  const northEast = L.latLng(7.255500, 80.593528);
  const bounds = L.latLngBounds(southWest, northEast);

  map = L.map(mapDivId, {
    maxZoom: 22,
    minZoom: 18,
    maxBounds: bounds,
    maxBoundsViscosity: 1.0,
  }).setView([7.253750, 80.592028], 19);

  // Custom pane for routes
  map.createPane("routePane");
  map.getPane("routePane").style.zIndex = 650;

  // Tile layer
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 22,
  }).addTo(map);

  // Example marker
  L.marker([7.253891, 80.592230]).addTo(map);

  // Load SVG overlay
  fetch("http://localhost:3000/map")
    .then((res) => res.text())
    .then((svgText) => {
      const parser = new DOMParser();
      const svgDoc = parser.parseFromString(svgText, "image/svg+xml");
      const svgElement = svgDoc.documentElement;
      L.svgOverlay(svgElement, bounds).addTo(map);
    })
    .catch((err) => console.error("Error loading SVG:", err));

  map.fitBounds(bounds);
}

async function getRoute(userLatLng, dest) {
  try {
    const res = await fetch(
      `http://localhost:3000/routing?lat=${userLatLng[0]}&long=${userLatLng[1]}&dest=${dest}`
    );
    return res.json();
  } catch (err) {
    console.error("Error fetching route:", err);
    return null;
  }
}

function drawRoute(result) {
  if (!result || !map) return;

  L.circleMarker(result.snappedAt, {
    radius: 8,
    color: "orange",
    pane: "routePane",
  })
    .addTo(map)
    .bindTooltip("Snapped");

  L.polyline(result.routeCoords, {
    color: "red",
    weight: 5,
    pane: "routePane",
  }).addTo(map);

  L.circleMarker(result.routeCoords.at(-1), {
    radius: 7,
    color: "red",
    pane: "routePane",
  })
    .addTo(map)
    .bindTooltip("Destination");

  map.fitBounds(result.routeCoords);
}

const mapLib = { initMap, getRoute, drawRoute };
export default mapLib;
