import React, { useState, useEffect } from 'react';

interface MapPageTailwindProps {}

const MapPageTailwind: React.FC<MapPageTailwindProps> = () => {
  const [selectedBuilding, setSelectedBuilding] = useState<string>('');
  const [userLatLng, setUserLatLng] = useState<[number, number] | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string>('');

  useEffect(() => {
    const initializeMap = async () => {
      setIsLoading(true);
      try {
        // Import map module dynamically to avoid TypeScript issues
        const mapModule = await import('./utils/map_module') as any;
        
        // Initialize map
        mapModule.initMap('map');
        
        // Set up building click listener
        mapModule.addBuildingClickListner((buildingId: string) => {
          setSelectedBuilding(buildingId);
        });

        // Try to get GPS position
        try {
          const position = await mapModule.getGpsPosition();
          setUserLatLng(position);
          mapModule.drawMarker(position);
        } catch (gpsError) {
          console.warn('Could not get GPS position:', gpsError);
          setErrorMessage('GPS location not available. Map features may be limited.');
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to initialize map:', error);
        setErrorMessage('Failed to load map. Please refresh the page.');
        setIsLoading(false);
      }
    };

    initializeMap();
  }, []);

  const handleGetRouteToBuilding = async () => {
    if (!userLatLng || !selectedBuilding) {
      setErrorMessage('Please select a building and ensure GPS is available.');
      return;
    }

    try {
      const mapModule = await import('./utils/map_module') as any;
      const route = await mapModule.getRouteToBuilding(userLatLng, selectedBuilding);
      mapModule.drawRoute(route);
      setErrorMessage('');
    } catch (error) {
      console.error('Failed to get route:', error);
      setErrorMessage('Failed to calculate route. Please try again.');
    }
  };

  const handleClearError = () => {
    setErrorMessage('');
  };

  return (
    <div className="min-h-screen  p-6">
      {/* Header */}
      <div className="text-center mb-6">
        <h1 className="text-4xl font-bold text-white mb-2 drop-shadow-lg">
          Campus Map
        </h1>
        <p className="text-xl text-blue-100">
          Navigate the University of Peradeniya Campus
        </p>
      </div>

      {/* Controls Section */}
      <div className="max-w-6xl mx-auto mb-6">
        <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            {/* Selected Building Info */}
            <div className="flex-1 min-w-64">
              <label className="block text-sm font-medium text-white mb-2">
                Selected Building:
              </label>
              <div className="bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2 border border-white/30">
                <span className="text-white">
                  {selectedBuilding || 'Click on a building on the map'}
                </span>
              </div>
            </div>

            {/* GPS Status */}
            <div className="flex-1 min-w-64">
              <label className="block text-sm font-medium text-white mb-2">
                GPS Status:
              </label>
              <div className={`bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2 border border-white/30 flex items-center ${
                userLatLng ? 'text-green-300' : 'text-yellow-300'
              }`}>
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  userLatLng ? 'bg-green-400' : 'bg-yellow-400'
                }`}></div>
                <span>
                  {userLatLng ? 'Location detected' : 'Location unavailable'}
                </span>
              </div>
            </div>

            {/* Route Button */}
            <div className="flex-shrink-0">
              <button
                onClick={handleGetRouteToBuilding}
                disabled={!userLatLng || !selectedBuilding}
                className={`px-6 py-3 rounded-xl font-semibold text-white transition-all duration-200 ${
                  userLatLng && selectedBuilding
                    ? 'bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 shadow-lg hover:shadow-xl transform hover:scale-105'
                    : 'bg-gray-500/50 cursor-not-allowed'
                }`}
              >
                Get Route
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div className="max-w-6xl mx-auto mb-6">
          <div className="bg-red-500/20 backdrop-blur-md border border-red-400/30 rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-red-400 rounded-full mr-3"></div>
              <span className="text-red-100">{errorMessage}</span>
            </div>
            <button
              onClick={handleClearError}
              className="text-red-200 hover:text-white ml-4 text-sm underline"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Map Container */}
      <div className="max-w-6xl mx-auto">
        <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-6 shadow-2xl">
          <div className="relative">
            {isLoading && (
              <div className="absolute inset-0 z-10 bg-black/20 backdrop-blur-sm rounded-xl flex items-center justify-center">
                <div className="text-center">
                  <div className="w-8 h-8 border-4 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                  <p className="text-white">Loading map...</p>
                </div>
              </div>
            )}
            
            <div 
              id="map" 
              className="w-full h-96 md:h-[500px] lg:h-[600px] rounded-xl bg-gray-200/20 border border-white/30"
              style={{ minHeight: '400px' }}
            ></div>
          </div>
        </div>
      </div>

      {/* Map Instructions */}
      <div className="max-w-6xl mx-auto mt-6">
        <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-4">
          <h3 className="text-lg font-semibold text-white mb-3">How to use:</h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="flex items-start">
              <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center mr-3 mt-1">
                <span className="text-white text-sm font-bold">1</span>
              </div>
              <div>
                <p className="text-white font-medium">Click on Buildings</p>
                <p className="text-blue-200 text-sm">Tap any building on the map to select it</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center mr-3 mt-1">
                <span className="text-white text-sm font-bold">2</span>
              </div>
              <div>
                <p className="text-white font-medium">Enable Location</p>
                <p className="text-green-200 text-sm">Allow location access for routing</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center mr-3 mt-1">
                <span className="text-white text-sm font-bold">3</span>
              </div>
              <div>
                <p className="text-white font-medium">Get Directions</p>
                <p className="text-purple-200 text-sm">Click "Get Route" for navigation</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MapPageTailwind;
