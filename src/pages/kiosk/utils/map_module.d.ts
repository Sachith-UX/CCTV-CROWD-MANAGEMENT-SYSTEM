// TypeScript declarations for map_module.js

export interface MapLocation {
  name: string;
  x: number;
  y: number;
  description?: string;
}

export interface MapModule {
  initializeMap: (containerId: string) => void;
  addMarker: (location: MapLocation) => void;
  clearMarkers: () => void;
  setMapSize: (width: number, height: number) => void;
}

declare const mapModule: MapModule;
export default mapModule;
