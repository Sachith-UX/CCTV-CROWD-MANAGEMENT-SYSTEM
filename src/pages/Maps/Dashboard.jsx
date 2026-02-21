import React, { useEffect, useMemo, useRef, useState } from 'react';
import './Dashboard.css';
import MapComponent from './Map.jsx';
import { searchDatabase } from './search.js';

const categories = {
  Exhibits: '#2563eb',
  Amenities: '#16a34a',
  Emergency: '#ef4444'
};

const zoneDropdown = [
  {
    key: 'zone1',
    label: 'Zone 1',
    subzones: [
      { key: 'zone1-all', label: 'All Zone 1' },
      { key: 'zone1-subA', label: 'Subzone A' },
      { key: 'zone1-subB', label: 'Subzone B' }
    ]
  },
  {
    key: 'zone2',
    label: 'Zone 2',
    subzones: [
      { key: 'zone2-all', label: 'All Zone 2' },
      { key: 'zone2-subA', label: 'Subzone A' }
    ]
  },
  {
    key: 'zone3',
    label: 'Zone 3',
    subzones: [
      { key: 'zone3-all', label: 'All Zone 3' }
    ]
  }
];

const allSubzones = [
  { key: 'zone1-subA', label: 'Zone 1 - Subzone A' },
  { key: 'zone1-subB', label: 'Zone 1 - Subzone B' },
  { key: 'zone2-subA', label: 'Zone 2 - Subzone A' },
  // Add more subzones as needed
  { key: 'zone1-all', label: 'All Zone 1' },
  { key: 'zone2-all', label: 'All Zone 2' },
  { key: 'zone3-all', label: 'All Zone 3' }
];




const BookmarkIcon = ({ filled = false }) => (
  <svg width="22" height="22" viewBox="0 0 22 22" style={{ verticalAlign: 'middle' }}>
    <path
      d="M6 3.5A2.5 2.5 0 0 1 8.5 1h5A2.5 2.5 0 0 1 16 3.5v15.2a.7.7 0 0 1-1.1.6l-4.4-2.7-4.4 2.7A.7.7 0 0 1 6 18.7V3.5z"
      fill={filled ? "#2563eb" : "none"}
      stroke="#2563eb"
      strokeWidth="1.5"
      strokeLinejoin="round"
    />
  </svg>
);

// Add a simple directions icon (SVG)
const DirectionsIcon = () => (
  <svg width="22" height="22" viewBox="0 0 22 22" style={{ verticalAlign: 'middle' }}>
    <path
      d="M11 2l8 8-8 8-8-8 8-8zm0 4v4h4"
      stroke="#2563eb"
      strokeWidth="1.5"
      fill="none"
      strokeLinejoin="round"
      strokeLinecap="round"
    />
  </svg>
);

// Add a locate me icon (SVG)
const LocateIcon = () => (
  <svg width="22" height="22" viewBox="0 0 22 22" style={{ verticalAlign: 'middle' }}>
    <circle cx="11" cy="11" r="3" fill="currentColor"/>
    <path d="M11 1v4M11 17v4M21 11h-4M5 11H1" stroke="currentColor" strokeWidth="2" fill="none"/>
  </svg>
);

// Search icon component (clickable)
// Custom SVG Bell Icon for Notification
const BellIcon = () => (
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ verticalAlign: 'middle' }}>
    <path d="M11 20c1.104 0 2-.896 2-2h-4c0 1.104.896 2 2 2zm6-6v-5c0-3.07-2.13-5.64-5-6.32V2a1 1 0 1 0-2 0v.68C5.13 3.36 3 5.92 3 9v5l-1 1v1h16v-1l-1-1z" stroke="#2563eb" strokeWidth="1.5" fill="#fff"/>
  </svg>
);
const SearchIconBtn = ({ onClick }) => (
  <button
    className="map-search-icon-btn"
    type="button"
    aria-label="Search"
    onClick={onClick}
    tabIndex={0}
    style={{
      position: 'absolute',
      right: '40px',
      top: '50%',
      transform: 'translateY(-50%)',
      background: 'none',
      border: 'none',
      padding: 0,
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}
  >
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
      <circle cx="10" cy="10" r="7" stroke="#2563eb" strokeWidth="2"/>
      <line x1="16" y1="16" x2="21" y2="21" stroke="#2563eb" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  </button>
);

// --- Modularized Subcomponents ---
function MapSearchBar({ 
  searchQuery, 
  setSearchQuery, 
  allResults, 
  handleSelectResult, 
  categories, 
  isSearching, 
  hasSelectedResult, 
  isSelecting,
  buildingRelatedResults,
  showBuildingResults
}) {
  return (
    <div className="map-search" style={{ position: 'relative' }}>
      <input
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder="Search exhibits..."
        className="map-search-input"
        style={{ paddingRight: '48px' }}
        onKeyDown={e => {
          if (e.key === 'Enter' && searchQuery) {
            // Optionally trigger search logic here
          }
        }}
      />
      <SearchIconBtn onClick={() => {
        document.querySelector('.map-search-input')?.focus();
      }} />
      {searchQuery && (
        <button className="map-search-clear" onClick={() => {
          setSearchQuery('');
          setShowBuildingResults(false);
          setBuildingRelatedResults([]);
          setHasSelectedResult(false);
          setIsSelecting(false);
        }} aria-label="Clear search">×</button>
      )}
      
      {/* Show building-related results when a building is selected */}
      {searchQuery && showBuildingResults && buildingRelatedResults.length > 0 && (
        <div className="map-search-results">
          <div style={{ padding: '8px 12px', fontSize: '12px', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
            Related items in {searchQuery}:
          </div>
          {buildingRelatedResults.slice(0, 8).map(item => (
            <button key={item.id} className="map-search-result" onClick={() => handleSelectResult(item)}>
              <span className="map-result-dot" style={{ backgroundColor: categories[item.category] || '#6b7280' }} />
              <div className="map-result-text">
                <div className="map-result-title">{item.name}</div>
                <div className="map-result-sub">
                  {item.type === 'building' ? 'Building' : 
                   item.type === 'exhibit' ? 'Exhibit' : 
                   item.type === 'amenity' ? 'Amenity' : item.category}
                  {item.buildingName && ` • ${item.buildingName}`}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
      
      {/* Show regular search results when not showing building results */}
      {searchQuery && !hasSelectedResult && !isSelecting && !showBuildingResults && (isSearching || allResults.length > 0) && (
        <div className="map-search-results">
          {isSearching ? (
            <div style={{ padding: '12px', textAlign: 'center', color: '#6b7280' }}>
              Searching...
            </div>
          ) : allResults.length > 0 ? (
            allResults.slice(0, 8).map(item => (
            <button key={item.id} className="map-search-result" onClick={() => handleSelectResult(item)}>
                <span className="map-result-dot" style={{ backgroundColor: categories[item.category] || '#6b7280' }} />
              <div className="map-result-text">
                <div className="map-result-title">{item.name}</div>
                  <div className="map-result-sub">
                    {item.type === 'building' ? 'Building' : 
                     item.type === 'exhibit' ? 'Exhibit' : 
                     item.type === 'amenity' ? 'Amenity' : item.category}
                    {item.buildingName && ` • ${item.buildingName}`}
                  </div>
              </div>
            </button>
            ))
          ) : null}
        </div>
      )}
    </div>
  );
}

function ZoneFilterPopup({
  showZoneFilter, setShowZoneFilter,
  selectedZone, setSelectedZone,
  selectedSubzone, setSelectedSubzone,
  zoneDropdown, allSubzones
}) {
  if (!showZoneFilter) return null;
  return (
    <div className="dashboard-zone-filter-popup" style={{ position: 'absolute', top: '110%', left: '0px', minWidth: 260, background: '#fff', borderRadius: 16, boxShadow: '0 8px 32px rgba(37,99,235,0.12)', padding: '18px 18px 12px 18px', zIndex: 9999 }}>
      <div style={{ fontWeight: 600, color: '#2563eb', marginBottom: 8 }}>Zone/Subzone Filters</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <select
          className="dashboard-zone-dropdown-select"
          value={selectedZone}
          onChange={e => {
            setSelectedZone(e.target.value);
            setSelectedSubzone('all');
          }}
        >
          <option value="all">All Zones</option>
          {zoneDropdown.map(zone => (
            <option key={zone.key} value={zone.key}>{zone.label}</option>
          ))}
        </select>
        <select
          className="dashboard-zone-dropdown-select"
          value={selectedSubzone}
          onChange={e => setSelectedSubzone(e.target.value)}
        >
          <option value="all">All Subzones</option>
          {selectedZone === 'all'
            ? allSubzones.map(sub => (
                <option key={sub.key} value={sub.key}>{sub.label}</option>
              ))
            : (zoneDropdown.find(z => z.key === selectedZone)?.subzones || []).map(sub => (
                <option key={sub.key} value={sub.key}>{sub.label}</option>
              ))
          }
        </select>
      </div>
      <button
        className="dashboard-zone-filter-close"
        type="button"
        onClick={() => setShowZoneFilter(false)}
        style={{ marginTop: 12, background: '#2563eb', color: '#fff', borderRadius: '8px', border: 'none', padding: '8px 18px', fontWeight: 600, cursor: 'pointer' }}
      >Done</button>
    </div>
  );
}

function NotificationPanel({ show, notification, onClose }) {
  if (!show) return null;
  return (
    <div className="dashboard-notification-panel" style={{ position: 'absolute', top: '110%', right: 0, minWidth: 260, background: '#fff', borderRadius: 16, boxShadow: '0 8px 32px rgba(37,99,235,0.12)', padding: '18px 18px 12px 18px', zIndex: 9999 }}>
      <div style={{ fontWeight: 600, color: '#2563eb', marginBottom: 8 }}>Notifications</div>
      {notification ? (
        <div className="dashboard-bookmark-notification">
          <span role="img" aria-label="notification" style={{ marginRight: 6 }}>🔔</span>
          <span>
            {(() => {
              const now = Date.now();
              const start = typeof notification.startTime === 'number' ? notification.startTime : new Date(notification.startTime).getTime();
              if (now < start) {
                return <>Upcoming: <strong>{notification.name}</strong> at {new Date(start).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</>;
              } else {
                return <>Event <strong>{notification.name}</strong> is starting now!</>;
              }
            })()}
          </span>
        </div>
      ) : (
        <div style={{ color: '#6b7280', fontSize: '0.95rem' }}>No notifications.</div>
      )}
      <button
        style={{ marginTop: 12, background: '#2563eb', color: '#fff', borderRadius: '8px', border: 'none', padding: '8px 18px', fontWeight: 600, cursor: 'pointer' }}
        onClick={onClose}
      >Close</button>
    </div>
  );
}

function MapLegend({ categories, activeCategories, handleLegendFilter }) {
  return (
    <div className="map-legend">
      <div className="map-legend-title">Legend (Click to Filter)</div>
      <div className="map-legend-items">
        {Object.keys(categories).map(cat => (
          <button
            key={cat}
            className={`map-legend-item${activeCategories[cat] ? ' active' : ''}`}
            style={{
              background: activeCategories[cat] ? '#e3f2fd' : 'transparent',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '11px',
              color: activeCategories[cat] ? '#1976d2' : '#6b7280',
              padding: '4px 8px',
              borderRadius: '8px'
            }}
            onClick={() => handleLegendFilter(cat)}
            type="button"
          >
            <span className="map-legend-dot" style={{ backgroundColor: categories[cat] }}></span>
            <span>{cat}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function BookmarkSidebar({ bookmarkedPoints, categories, setSelectedPoint }) {
  return (
    <div className="dashboard-bookmarks-list">
      {bookmarkedPoints.length === 0 ? (
        <div className="dashboard-bookmarks-empty">No bookmarks yet.</div>
      ) : (
        bookmarkedPoints.map(bookmark => (
          <button
            key={bookmark.id}
            className="dashboard-bookmark-item"
            onClick={() => {
              // Show building details when bookmark is clicked
              if (window.showBuildingInfo) {
                window.showBuildingInfo(bookmark.id, bookmark.name);
              }
            }}
          >
            <span className="dashboard-bookmark-dot" style={{ backgroundColor: '#f59e0b' }} />
            <span className="dashboard-bookmark-name">{bookmark.name}</span>
            <span className="dashboard-bookmark-icon">
              <BookmarkIcon filled />
            </span>
          </button>
        ))
      )}
    </div>
  );
}

const Dashboard = () => {
  const mapRef = useRef(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategories, setActiveCategories] = useState({ Exhibits: true, Amenities: true, Emergency: true });
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [isLocating, setIsLocating] = useState(false);
  const [bookmarks, setBookmarks] = useState([]);
  const [sidebarTab, setSidebarTab] = useState('main'); // 'main' or 'bookmarks'
  const [showZoneFilter, setShowZoneFilter] = useState(false);
  const [selectedZone, setSelectedZone] = useState('all');
  const [selectedSubzone, setSelectedSubzone] = useState('all');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSelectedResult, setHasSelectedResult] = useState(false);
  const [isSelecting, setIsSelecting] = useState(false);
  const [buildingRelatedResults, setBuildingRelatedResults] = useState([]);
  const [showBuildingResults, setShowBuildingResults] = useState(false);

  const visiblePoints = useMemo(() => {
    // Use search results when available
    if (searchQuery.trim() && searchResults.length > 0) {
      return searchResults.filter(item => {
        const category = item.type === 'building' ? 'Building' : 
                        item.type === 'exhibit' ? 'Exhibits' : 
                        item.type === 'amenity' ? 'Amenities' : item.category;
        return activeCategories[category];
      });
    }
    
    // Return empty array when no search results
    return [];
  }, [searchQuery, searchResults, activeCategories, selectedZone, selectedSubzone]);

  // Search function using API
  const performSearch = async (query, category, zone, subzone) => {
    if (!query || query.trim() === '') {
      setSearchResults([]);
      return;
    }
    
    setIsSearching(true);
    try {
      const results = await searchDatabase(query, { category, zone, subzone });
      setSearchResults(results);
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  // Function to fetch building-related items
  const fetchBuildingRelatedItems = async (buildingId, buildingName) => {
    try {
      // Search for all items in this building by using the building name as query
      const results = await searchDatabase(buildingName, { category: 'all', zone: 'all', subzone: 'all' });
      
      // Filter results to only include items from the same building
      const buildingItems = results.filter(item => 
        item.buildingId === buildingId || 
        item.buildingName === buildingName ||
        item.name === buildingName
      );
      
      setBuildingRelatedResults(buildingItems);
      setShowBuildingResults(true);
    } catch (error) {
      console.error('Failed to fetch building related items:', error);
      setBuildingRelatedResults([]);
    }
  };

  // Debounced search effect
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.trim()) {
        setHasSelectedResult(false); // Reset selection flag when user types
        setIsSelecting(false); // Reset selecting flag when user types
        console.log('User typing, resetting selection flag');
        performSearch(searchQuery, 'all', selectedZone, selectedSubzone);
      } else {
        setSearchResults([]);
        setIsSearching(false);
        setHasSelectedResult(false);
        setIsSelecting(false);
        setShowBuildingResults(false);
        setBuildingRelatedResults([]);
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timeoutId);
  }, [searchQuery, selectedZone, selectedSubzone, hasSelectedResult, showBuildingResults]);

  // Auto-select first result when there's only one result
  useEffect(() => {
    if (searchResults.length === 1 && !hasSelectedResult && !isSelecting && searchQuery.trim()) {
      // Small delay to ensure the search is complete
      const autoSelectTimeout = setTimeout(() => {
        console.log('Auto-selecting first result:', searchResults[0]);
        handleSelectResult(searchResults[0]);
      }, 100);
      
      return () => clearTimeout(autoSelectTimeout);
    }
  }, [searchResults, hasSelectedResult, isSelecting, searchQuery]);

  // Debug effect to monitor state changes
  useEffect(() => {
    console.log('State changed - searchQuery:', searchQuery, 'hasSelectedResult:', hasSelectedResult, 'isSelecting:', isSelecting, 'isSearching:', isSearching, 'allResults.length:', searchResults.length);
  }, [searchQuery, hasSelectedResult, isSelecting, isSearching, searchResults.length]);

  const allResults = searchResults;


  // Map initialization is now handled by MapComponent

  const handleSelectResult = (item) => {
    // Immediately hide dropdown by setting all flags
    console.log('handleSelectResult called, hiding dropdown immediately');
    setIsSelecting(true);
    setHasSelectedResult(true);
    setSearchResults([]);
    setIsSearching(false);
    
    // Convert API result to the format expected by the rest of the app
    const point = {
      id: item.id,
      name: item.name,
      category: item.type === 'building' ? 'Building' : 
                item.type === 'exhibit' ? 'Exhibits' : 
                item.type === 'amenity' ? 'Amenities' : item.category,
      x: item.coordinates ? item.coordinates[0] * 100 : 200, // Convert lat to approximate x
      y: item.coordinates ? item.coordinates[1] * 100 : 200, // Convert lng to approximate y
      description: item.description,
      buildingId: item.buildingId,
      buildingName: item.buildingName,
      svgBuildingId: item.svgBuildingId,
      type: item.type
    };
    
    setSelectedPoint(point);
    setSearchQuery(item.name);
    
    // Only highlight the building, don't show popup
    if (item.svgBuildingId) {
      // Use highlightBuilding from Map component via window object
      if (window.highlightBuilding) {
        window.highlightBuilding(item.svgBuildingId);
      }
    }
    
    centerOnPoint(point);
    
    // If this is a building selection, fetch and show related items from the same building
    if (item.type === 'building' && item.buildingId) {
      fetchBuildingRelatedItems(item.buildingId, item.name);
    } else {
      // If selecting a non-building item, hide building results
      setShowBuildingResults(false);
      setBuildingRelatedResults([]);
    }
  };

  const toggleCategory = (cat) => {
    setActiveCategories(prev => ({ ...prev, [cat]: !prev[cat] }));
  };

  const handleLegendFilter = (cat) => {
    setActiveCategories(prev => ({
      ...prev,
      [cat]: !prev[cat]
    }));
  };

  // Quick filter presets for easy viewing
  const quickFilter = (key) => {
    if (key === 'All') {
      setActiveCategories({ Exhibits: true, Amenities: true, Emergency: true });
    } else if (key === 'Amenities') {
      setActiveCategories({ Exhibits: false, Amenities: true, Emergency: false });
    } else if (key === 'Emergency') {
      setActiveCategories({ Exhibits: false, Amenities: false, Emergency: true });
    }
  };

  // Map styling is now handled by MapComponent



  // Bookmark logic
  const isBookmarked = selectedPoint && bookmarks.includes(selectedPoint.id);
  const toggleBookmark = () => {
    if (!selectedPoint) return;
    setBookmarks(prev =>
      prev.includes(selectedPoint.id)
        ? prev.filter(id => id !== selectedPoint.id)
        : [...prev, selectedPoint.id]
    );
  };

  // Add bookmark function for Map component
  const addBookmark = (bookmark) => {
    console.log('addBookmark called with:', bookmark);
    if (!bookmark || !bookmark.id) {
      console.log('Invalid bookmark data');
      return;
    }
    
    setBookmarks(prev => {
      console.log('Current bookmarks:', prev);
      // Check if already bookmarked
      if (prev.some(b => b.id === bookmark.id)) {
        console.log('Already bookmarked');
        return prev; // Already bookmarked
      }
      // Add new bookmark
      const newBookmarks = [...prev, bookmark];
      console.log('New bookmarks:', newBookmarks);
      return newBookmarks;
    });
  };

  // Memo for bookmarked points
  const bookmarkedPoints = useMemo(
    () => {
      console.log('bookmarkedPoints useMemo - bookmarks:', bookmarks);
      // If bookmarks are objects (new format), use them directly
      if (bookmarks.length > 0 && typeof bookmarks[0] === 'object') {
        console.log('Using new format bookmarks:', bookmarks);
        return bookmarks;
      }
      // If bookmarks are IDs (old format), return empty array since we don't have mockPoints
      console.log('Old format bookmarks detected, but no mockPoints available');
      return [];
    },
    [bookmarks]
  );

  // Expose addBookmark function globally for Map component
  useEffect(() => {
    window.addBookmark = addBookmark;
    return () => {
      delete window.addBookmark;
    };
  }, []);


  // Center map on a specific point (used in search results)
  const centerOnPoint = (pt) => {
    if (!pt || !map) return;
    // You can enhance this later to center the Leaflet map on the point
    // The popup is now handled by the new unified popup system
  };

  const locateMe = async () => {
    setIsLocating(true);
    setTimeout(() => setIsLocating(false), 1000);
    // UI only for now
  };

  // Notification state
  const [notification, setNotification] = useState(null);
  const [showNotificationPopup, setShowNotificationPopup] = useState(false);
  const [showNotificationPanel, setShowNotificationPanel] = useState(false);

  // Helper: get next notification event (10 min before or at start)
  const getNextNotification = () => {
    const now = Date.now();
    // Only for bookmarked points with startTime
    if (bookmarkedPoints.length === 0) return null; // Only enable notification if there are bookmarks
    const events = bookmarkedPoints
      .filter(p => p.startTime)
      .map(p => {
        const start = typeof p.startTime === 'number' ? p.startTime : new Date(p.startTime).getTime();
        return { ...p, start };
      });
    for (const ev of events) {
      const minBefore = ev.start - 10 * 60 * 1000;
      if (
        (now >= minBefore && now < ev.start) || // 10 min before
        (now >= ev.start && now < ev.start + 60 * 1000) // at start, show for 1 min
      ) {
        return ev;
      }
    }
    return null;
  };

  // Notification effect: show popup only once per event
  useEffect(() => {
    let notifiedEventId = null;
    const checkNotification = () => {
      // Only enable notification if there are bookmarks
      if (bookmarkedPoints.length === 0) {
        setShowNotificationPopup(false);
        setNotification(null);
        return;
      }
      const next = getNextNotification();
      if (next && notifiedEventId !== next.id) {
        setNotification(next);
        setShowNotificationPopup(true);
        notifiedEventId = next.id;
        setTimeout(() => setShowNotificationPopup(false), 60000); // auto-hide after 1 min
      }
    };
    const interval = setInterval(checkNotification, 10000); // check every 10s
    checkNotification();
    return () => clearInterval(interval);
    // eslint-disable-next-line
  }, [bookmarkedPoints]);

  const [isFullscreen, setIsFullscreen] = useState(false);
  const handleFullscreen = () => {
    const mapElement = document.getElementById('map');
    if (!mapElement) return;
    if (!isFullscreen) {
      if (mapElement.requestFullscreen) mapElement.requestFullscreen();
      else if (mapElement.webkitRequestFullscreen) mapElement.webkitRequestFullscreen();
      else if (mapElement.mozRequestFullScreen) mapElement.mozRequestFullScreen();
      else if (mapElement.msRequestFullscreen) mapElement.msRequestFullscreen();
    } else {
      if (document.exitFullscreen) document.exitFullscreen();
      else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      else if (document.mozCancelFullScreen) document.mozCancelFullScreen();
      else if (document.msExitFullscreen) document.msExitFullscreen();
    }
  };
  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(!!(document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement));
    };
    document.addEventListener('fullscreenchange', onFsChange);
    document.addEventListener('webkitfullscreenchange', onFsChange);
    document.addEventListener('mozfullscreenchange', onFsChange);
    document.addEventListener('MSFullscreenChange', onFsChange);
    return () => {
      document.removeEventListener('fullscreenchange', onFsChange);
      document.removeEventListener('webkitfullscreenchange', onFsChange);
      document.removeEventListener('mozfullscreenchange', onFsChange);
      document.removeEventListener('MSFullscreenChange', onFsChange);
    };
  }, []);

  return (
    <div className="dashboard-container with-bottom-padding">
      <div className="map-page">
        <div className="map-header">
          <div className="dashboard-search-center">
            <MapSearchBar
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              allResults={allResults}
              handleSelectResult={handleSelectResult}
              categories={categories}
              isSearching={isSearching}
              hasSelectedResult={hasSelectedResult}
              isSelecting={isSelecting}
              buildingRelatedResults={buildingRelatedResults}
              showBuildingResults={showBuildingResults}
            />
            <div className="dashboard-zone-filter-btn-wrap" style={{ position: 'relative' }}>
              <button
                className="dashboard-zone-filter-btn"
                type="button"
                onClick={() => setShowZoneFilter(v => !v)}
                aria-label="Zone/Subzone Filters"
                style={{
                  width: 44,
                  height: 44,
                  padding: 0,
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 22
                }}
              >
                <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                  <rect x="6" y="7" width="16" height="2.5" rx="1.25" fill="#2563eb"/>
                  <rect x="6" y="13" width="10" height="2.5" rx="1.25" fill="#2563eb"/>
                  <rect x="6" y="19" width="13" height="2.5" rx="1.25" fill="#2563eb"/>
                </svg>
              </button>
              <ZoneFilterPopup
                showZoneFilter={showZoneFilter}
                setShowZoneFilter={setShowZoneFilter}
                selectedZone={selectedZone}
                setSelectedZone={setSelectedZone}
                selectedSubzone={selectedSubzone}
                setSelectedSubzone={setSelectedSubzone}
                zoneDropdown={zoneDropdown}
                allSubzones={allSubzones}
              />
            </div>
            <div className="dashboard-notification-btn-wrap" style={{ position: 'relative', marginLeft: 8 }}>
              <button
                className="dashboard-zone-filter-btn"
                type="button"
                aria-label="Notifications"
                style={{ width: 44, height: 44, padding: 0, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22 }}
                onClick={() => setShowNotificationPanel(v => !v)}
              >
                <BellIcon />
              </button>
              <NotificationPanel
                show={showNotificationPanel}
                notification={notification}
                onClose={() => setShowNotificationPanel(false)}
              />
            </div>
          </div>
        </div>
        <div className="map-layout">
          {/* Left Sidebar */}
          <div className="map-sidebar">
            <div>
              <div className="dashboard-filter-card">
                <button
                  className="dashboard-filter-btn"
                  onClick={() => setActiveCategories(prev => {
                    const allActive = Object.values(prev).every(Boolean);
                    return {
                      Exhibits: !allActive,
                      Amenities: !allActive,
                      Emergency: !allActive
                    };
                  })}
                >
                  {Object.values(activeCategories).every(Boolean) ? 'Hide All Categories' : 'Show All Categories'}
                </button>
              </div>
              <MapLegend
                categories={categories}
                activeCategories={activeCategories}
                handleLegendFilter={handleLegendFilter}
              />
            </div>
          </div>
          {/* Main Map */}
          <div className="map-main">
            <div className="map-card">
              <div className="map-viewport">
                <MapComponent />
                {/* Custom Controls */}
                <div className="map-controls">
                  <button className={`map-ctrl ${isLocating ? 'loading' : ''}`} onClick={locateMe} aria-label="Locate me">
                    <LocateIcon />
                  </button>
                  <button className="map-ctrl" onClick={handleFullscreen} aria-label="Fullscreen" title="Fullscreen">
                    {isFullscreen ? (
                      <span role="img" aria-label="Exit Fullscreen">🗗</span>
                    ) : (
                      <span role="img" aria-label="Fullscreen">🗖</span>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
          {/* Bookmark Sidebar as a block */}
          <div className="map-bookmark-sidebar-block" style={{
            flex: '0 0 280px',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
            background: '#fff',
            borderRadius: '20px',
            boxShadow: '8px 8px 16px rgba(163, 177, 198, 0.15), -8px -8px 16px rgba(255,255,255,0.8)',
            padding: '20px',
            marginLeft: '1rem',
            minHeight: '400px'
          }}>
            <div className="dashboard-sidebar-tabs">
              <button
                className={`dashboard-sidebar-tab${sidebarTab === 'bookmarks' ? ' active' : ''}`}
                onClick={() => setSidebarTab('bookmarks')}
                type="button"
              >
                Bookmarks
                {bookmarks.length > 0 && (
                  <span className="dashboard-bookmark-count">{bookmarks.length}</span>
                )}
              </button>
            </div>
            <BookmarkSidebar
              bookmarkedPoints={bookmarkedPoints}
              categories={categories}
              setSelectedPoint={setSelectedPoint}
            />
          </div>
        </div>
      </div>
      {/* Notification popup: only show once, 10 min before or at start time */}
      {showNotificationPopup && notification && (
        <div className="dashboard-notification-popup">
          <span role="img" aria-label="notification" style={{ fontSize: 22 }}>🔔</span>
          <div>
            <div style={{ fontWeight: 600, color: '#2563eb' }}>{notification.name}</div>
            <div style={{ fontSize: 14, color: '#374151' }}>
              {(() => {
                const now = Date.now();
                const start = typeof notification.startTime === 'number' ? notification.startTime : new Date(notification.startTime).getTime();
                if (now < start) {
                  return `Starts at ${new Date(start).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
                } else {
                  return `Event is starting now!`;
                }
              })()}
            </div>
          </div>
          <button
            style={{
              marginLeft: 'auto',
              background: 'none',
              border: 'none',
              fontSize: 22,
              cursor: 'pointer',
              color: '#2563eb'
            }}
            aria-label="Close notification"
            onClick={() => setShowNotificationPopup(false)}
          >×</button>
        </div>
      )}
    </div>
  );
};

export default Dashboard;