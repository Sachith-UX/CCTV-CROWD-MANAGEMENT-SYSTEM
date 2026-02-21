import { useEffect, useRef, useState } from "react";
import "leaflet/dist/leaflet.css";
import './Map.css';

import {map, initMap, getRouteToNode, getRouteToBuilding, drawRoute, addBuildingClickListner, getGpsPosition, drawMarker } from "./map_module.js";  

function MapComponent() {
  const mapRef = useRef(null);
  const [showBuildingPopup, setShowBuildingPopup] = useState(false);
  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [buildingDetails, setBuildingDetails] = useState(null);

  useEffect(() => {
    if (mapRef.current) return; // prevent re-init

    // Initialize the map
    initMap('map');
    mapRef.current = map

    // Apply minimal enhancements
    enhanceMapFunctionality();

    // Get user location immediately and make it visible
    getGpsPosition()
    .then((pos) => {
      console.log("GPS position obtained:", pos);
      
      // Always draw marker when GPS location is obtained
      if (pos && pos.length === 2) {
        drawMarker(pos);
        console.log("User location marker drawn at startup");
      }
    })
    .catch((e)=>{
      console.log("GPS failed at startup:", e);
      // Optionally show a default location or retry
    });

    // Set up building click listener
    addBuildingClickListner((buildingId) => {
      console.log("Building clicked:", buildingId);
      
      // Show popup with building info
      showBuildingInfo(buildingId);
    });

  }, []);

  // Handle window resize for responsive map
  useEffect(() => {
    const handleResize = () => {
      if (mapRef.current) {
        setTimeout(() => {
          mapRef.current.invalidateSize();
        }, 100);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const mapStyle = { 
    height: "740px", 
    width: "840px",
    borderRadius: "12px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
    transition: "all 0.3s ease"
  };

  // Function to enhance map functionality after initialization
  const enhanceMapFunctionality = () => {
    if (!mapRef.current) return;

    const map = mapRef.current;
    
    // Only add unique enhancements not handled by map_module.js
    // Create additional panes for better layering
    if (!map.getPane('svgPane')) {
      map.createPane('svgPane');
      map.getPane('svgPane').style.zIndex = 600;
    }
    
    console.log('Map functionality enhanced');
  };

  // Building highlight functions (UI-related functionality)
  const highlightBuilding = (svgBuildingId) => {
    if (!svgBuildingId) return;
    
    // Clear any existing highlights
    clearBuildingHighlight();
    
    // Find the SVG element and highlight it
    const svgElement = document.getElementById(svgBuildingId);
    if (svgElement) {
      svgElement.style.fill = '#4A90E2'; // Blue highlight
      svgElement.style.stroke = '#0066CC';
      svgElement.style.strokeWidth = '3px';
      svgElement.style.filter = 'drop-shadow(0 0 8px rgba(0, 102, 204, 0.8))';
    }
  };

  const clearBuildingHighlight = () => {
    // Reset all building styles
    const buildings = document.querySelectorAll('[id^="b"]');
    buildings.forEach(building => {
      building.style.fill = '';
      building.style.stroke = '';
      building.style.strokeWidth = '';
      building.style.filter = '';
    });
  };




  const handlePopupBackgroundClick = (e) => {
    if (e.target.classList.contains('building-popup')) {
      setShowBuildingPopup(false);
      setSelectedBuilding(null);
      setBuildingDetails(null);
    }
  };

  // Function to convert SVG building ID to numeric building data ID
  const getBuildingDataId = (svgBuildingId) => {
    const buildingMapping = {
      'b11': 1,  // Main Exhibition Hall
      'b32': 2,  // Electrical Engineering Pavilion  
      'b33': 3,  // Computer Science Lab
      'b34': 4,  // Civil Engineering Center
      'b31': 5,  // Biomedical Lab
      'b16': 6,  // Mechanical Engineering Workshop
      'b28': 7,  // Chemical Engineering Lab
      'b13': 8,  // Aerospace Engineering Center
      'b30': 9,  // Food Court & Rest Area
      'b26': 10  // Information Desk
    };
    return buildingMapping[svgBuildingId] || null;
  };

  const fetchBuildingDetails = async (buildingId) => {
    try {
      // Convert SVG building ID to numeric building data ID
      const dataId = getBuildingDataId(buildingId);
      if (!dataId) {
        console.error('No building data found for SVG ID:', buildingId);
        return null;
      }
      
      const response = await fetch(`http://localhost:3000/api/building/${dataId}`);
      if (response.ok) {
        const details = await response.json();
        return details;
      }
    } catch (error) {
      console.error('Error fetching building details:', error);
    }
    return null;
  };

  const showBuildingInfo = async (buildingId, buildingName = null) => {
    const details = await fetchBuildingDetails(buildingId);
    setSelectedBuilding(buildingId);
    setBuildingDetails(details);
    setShowBuildingPopup(true);
    
    // Highlight the building - buildingId could be either SVG ID or numeric ID
    let svgBuildingId = buildingId;
    let numericBuildingId = buildingId;
    
    // If buildingId is numeric (from search results), get the SVG ID
    if (typeof buildingId === 'number' || /^\d+$/.test(buildingId)) {
      const buildingMapping = await fetch('http://localhost:3000/api/building-mapping').then(r => r.json());
      svgBuildingId = buildingMapping[buildingId];
    } else {
      // If buildingId is SVG ID (from direct clicks), get the numeric ID
      numericBuildingId = getBuildingDataId(buildingId);
    }
    
    if (svgBuildingId) {
      highlightBuilding(svgBuildingId);
    }
    
    // Note: Route will only be shown when user clicks Navigate button
  };

  // Expose functions to parent components
  useEffect(() => {
    window.showBuildingInfo = showBuildingInfo;
    window.highlightBuilding = highlightBuilding;
    window.clearBuildingHighlight = clearBuildingHighlight;
    // Import and expose navigation functions from map_module
    import('./map_module.js').then(module => {
      window.getGpsPosition = module.getGpsPosition;
      window.getRouteToBuilding = module.getRouteToBuilding;
      window.drawRoute = module.drawRoute;
    });
  }, []);

  return (
    <div>
      <div id="map" style={mapStyle}></div>
      
      {/* Building Details Bottom Sheet */}
      {showBuildingPopup && selectedBuilding && (
        <div
          className="building-bottom-sheet"
          style={{
            zIndex: 9999,
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            background: '#fff',
            borderTopLeftRadius: '20px',
            borderTopRightRadius: '20px',
            boxShadow: '0 -10px 32px rgba(0,0,0,0.18)',
            maxHeight: '70vh',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          {/* Drag Handle */}
          <div
            style={{
              width: '40px',
              height: '4px',
              background: '#e5e7eb',
              borderRadius: '2px',
              margin: '12px auto 8px auto',
              cursor: 'grab'
            }}
          />
          
          <div
            style={{
              padding: '0 1.5rem 1.5rem 1.5rem',
              overflow: 'auto',
              flex: 1
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 style={{ margin: 0, color: '#2563eb', fontSize: '1.5rem' }}>
                {buildingDetails?.name || `Building ${selectedBuilding}`}
              </h2>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <button
                  onClick={async () => {
                    // Hide the popup first so user can see the route clearly
                    setShowBuildingPopup(false);
                    
                    // Get user location and show route to building
                    try {
                      const userPos = await getGpsPosition();
                      if (userPos) {
                        const routeResult = await getRouteToBuilding(userPos, selectedBuilding);
                        if (routeResult) {
                          drawRoute(routeResult);
                          // Route is now displayed on the map - no alert needed
                        }
                      } else {
                        console.warn('Unable to get your location. Please enable GPS and try again.');
                      }
                    } catch (error) {
                      console.error('Navigation error:', error);
                      console.warn('Unable to get directions. Please check your GPS settings.');
                    }
                  }}
                  style={{
                    padding: '0.5rem 0.75rem',
                    backgroundColor: '#16a34a',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    fontWeight: '500'
                  }}
                >
                  🧭 Navigate
                </button>
                <button
                  onClick={() => {
                    // Add building to bookmarks
                    if (buildingDetails && selectedBuilding) {
                      const bookmark = {
                        id: selectedBuilding,
                        name: buildingDetails.name,
                        description: buildingDetails.description,
                        svgBuildingId: getBuildingDataId(selectedBuilding) ? selectedBuilding : null
                      };
                      
                      console.log('Adding bookmark:', bookmark);
                      
                      // Call the bookmark function from Dashboard
                      if (window.addBookmark) {
                        window.addBookmark(bookmark);
                      }
                    }
                  }}
                  style={{
                    padding: '0.5rem 0.75rem',
                    backgroundColor: '#f59e0b',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    fontWeight: '500'
                  }}
                >
                  🔖 Bookmark
                </button>
                <button
                  onClick={() => {
                    setShowBuildingPopup(false);
                    setSelectedBuilding(null);
                    setBuildingDetails(null);
                  }}
                  style={{
                    background: 'none',
                    border: 'none',
                    fontSize: '1.5rem',
                    cursor: 'pointer',
                    color: '#6b7280'
                  }}
                >
                  ×
                </button>
              </div>
            </div>
            
            {buildingDetails && (
              <>
                <div style={{ marginBottom: '1rem' }}>
                  <p style={{ margin: 0, color: '#374151', lineHeight: '1.6' }}>
                    {buildingDetails.description || 'This building is part of the university campus.'}
                  </p>
                </div>
                
                {buildingDetails.exhibits && buildingDetails.exhibits.length > 0 && (
                  <div style={{ marginBottom: '1rem' }}>
                    <h3 style={{ margin: '0 0 0.5rem 0', color: '#374151', fontSize: '1.1rem' }}>
                      🎯 Exhibits & Activities
                    </h3>
                    <ul style={{ margin: 0, paddingLeft: '1.2rem', color: '#6b7280', fontSize: '0.9rem' }}>
                      {buildingDetails.exhibits.map((exhibit, index) => (
                        <li key={index} style={{ marginBottom: '0.3rem' }}>{exhibit}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                
                {buildingDetails.categories && buildingDetails.categories.length > 0 && (
                  <div style={{ marginBottom: '1rem' }}>
                    <h3 style={{ margin: '0 0 0.5rem 0', color: '#374151', fontSize: '1.1rem' }}>
                      🏷️ Categories
                    </h3>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                      {buildingDetails.categories.map((category, index) => (
                        <span key={index} style={{
                          padding: '0.25rem 0.5rem',
                          backgroundColor: '#e3f2fd',
                          color: '#2563eb',
                          borderRadius: '12px',
                          fontSize: '0.8rem',
                          fontWeight: '500'
                        }}>
                          {category}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
            
          </div>
        </div>
      )}
    </div>
  );
 
}

export default MapComponent;
