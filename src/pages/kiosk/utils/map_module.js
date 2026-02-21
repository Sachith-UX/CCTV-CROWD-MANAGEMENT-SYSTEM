import L from "leaflet";

let map;

function buildingClick(id) {
  console.log("Building clicked:", id);
  if (buildingClickListner.length === 0) {
    console.warn("No building click listeners registered.");
    return
  }
  buildingClickListner.forEach(fn => fn(id));
}

function initMap(map_div) {
  const southWest = L.latLng(7.252000, 80.590528);
  const northEast = L.latLng(7.255500, 80.593528);
  const bounds = L.latLngBounds(southWest, northEast);

  map = L.map(map_div, {
    maxZoom: 22,
    minZoom: 18,
    maxBounds: bounds,
    maxBoundsViscosity: 1.0
  }).setView([7.253750, 80.592028], 19);

  // Create custom panes
  map.createPane('routePane');
  map.getPane('routePane').style.zIndex = 650;

  // Add OpenStreetMap tile layer
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 22
  }).addTo(map);


  // Load SVG overlay
  fetch('http://localhost:3000/map')
    .then(res => res.text())
    .then(svgText => {
      const parser = new DOMParser();
      const svgDoc = parser.parseFromString(svgText, "image/svg+xml");
      const svgElement = svgDoc.documentElement;
      L.svgOverlay(svgElement, bounds).addTo(map);

      //Attach click listeners to SVG buildings
      const buildings = document.querySelectorAll('#_x3C_buildings_x3E_ .st4');
      buildings.forEach(building => {
          building.addEventListener('click', (event) => {
              buildingClick(event.target.id);
          });
      });
    })
    .then(() => {
      //connectSSE();
    })
    .catch(err => console.error('Error loading SVG:', err));

  map.fitBounds(bounds);


}

async function getRouteToNode(userLatLng, dest) {
  if(dest === undefined){
    console.log("undefined route")
  } else{
    console.log(dest);
  var result = await fetch(`http://localhost:3000/routing?lat=${userLatLng[0]}&long=${userLatLng[1]}&dest=${dest}`);
  return result.json();
  }
}

const buildings = {
  "b29": 39,
  "b10": 29,
  "b16": 81,
  "b34": 61,
  "b15": 17,
  "b14": 84,
  "b6": 30,
  "b13": 36,
  "b7": 90,
  "b12": 83,
  "b33": 77,
  "b32": 41,
  "b11": 32,
  "b18": 59,
  "b20":35,
  "b21": 56,
  "b28": 40,
  "b22": 57,
  "b30": 54,
  "b23": 38,
  "b24": 37,
  "b4": 50,
  "b2": 24,
  "b1": 48,


};

async function getRouteToBuilding(userLatLng, buildingId) {
  if (buildingId){
  console.log(buildings[buildingId]);
  var result = await getRouteToNode(userLatLng, buildings[buildingId]);
  return result;
  } else{
    return null;
  }
}


function drawRoute(result) {

  console.log(result)

  map.getPane('routePane').querySelectorAll('path, circle, polygon').forEach(el => el.remove());

  if (result) {

    L.circleMarker(result.snappedAt, { radius: 8, color: 'orange', pane: 'routePane' }).addTo(map).bindTooltip('Snapped');
    L.polyline(result.routeCoords, { color: 'red', weight: 5, pane: 'routePane' }).addTo(map);
    L.circleMarker(result.routeCoords.at(-1), { radius: 7, color: 'red', pane: 'routePane' }).addTo(map).bindTooltip('end');
    map.fitBounds(result.routeCoords);
  }

}

function drawMarker(latLng) {
  if(latLng){
    L.circleMarker(latLng, { radius: 8, color: 'blue', pane: 'routePane' }).addTo(map);
  }
}



let buildingClickListner = [];

function addBuildingClickListner(listner) {
  buildingClickListner.push(listner);
}

async function getGpsPosition(){
  return new Promise((resolve, reject) => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((position) => {
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;
        console.log(`gps location: ${latitude}, ${longitude}`);
        resolve([latitude, longitude]);
      }, reject);
    } else {
      alert("Geolocation is not supported by this browser.");
    }
  })
}

export {map, initMap, getRouteToNode, getRouteToBuilding, drawRoute, addBuildingClickListner, getGpsPosition, drawMarker };





