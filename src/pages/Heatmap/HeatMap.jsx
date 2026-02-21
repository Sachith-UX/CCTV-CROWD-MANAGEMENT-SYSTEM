import React from 'react';

const HeatMap = () => {
  // Mock heat map data for frontend-only version
  const generateMockHeatMapData = () => {
    const data = [];
    for (let i = 0; i < 700; i++) { // 25x28 grid
      const intensity = Math.random();
      data.push({
        intensity: intensity,
        value: Math.floor(intensity * 100),
        zone: `Zone ${Math.floor(i / 28) + 1}-${(i % 28) + 1}`
      });
    }
    return data;
  };
  
  const heatMapData = generateMockHeatMapData();
  
  // Mock statistics
  const statistics = {
    totalOccupancy: 2847,
    averageUsage: 78
  };

  const getColor = (intensity) => {
    if (intensity < 0.2) return 'rgba(30, 64, 175, 0.4)'; // Blue (low) - transparent
    if (intensity < 0.4) return 'rgba(5, 150, 105, 0.5)'; // Green (low-medium) - transparent
    if (intensity < 0.6) return 'rgba(217, 119, 6, 0.6)'; // Orange (medium) - transparent
    if (intensity < 0.8) return 'rgba(220, 38, 38, 0.7)'; // Red (high) - transparent
    return 'rgba(124, 45, 18, 0.8)'; // Dark red (very high) - transparent
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 className="text-2xl">Real-Time Crowd Heat Map</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: 'rgba(30, 64, 175, 0.4)', border: '1px solid #1e40af' }}></div>
            <span>Low</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: 'rgba(5, 150, 105, 0.5)', border: '1px solid #059669' }}></div>
            <span>Medium</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: 'rgba(220, 38, 38, 0.7)', border: '1px solid #dc2626' }}></div>
            <span>High</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: 'rgba(124, 45, 18, 0.8)', border: '1px solid #7c2d12' }}></div>
            <span>Critical</span>
          </div>
        </div>
      </div>
      
      <div style={{
        position: 'relative',
        width: '100%',
        height: '500px',
        borderRadius: '8px',
        overflow: 'hidden',
        border: '2px solid #e5e7eb'
      }}>
        {/* Venue Map Background */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundImage: 'url("/images/EngEx Exhibition Draft Map.jpeg")',
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'center',
          backgroundColor: '#f0f9ff'
        }} />
        
        {/* Heat Map Overlay Grid */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          display: 'grid',
          gridTemplateColumns: 'repeat(25, 1fr)',
          gridTemplateRows: 'repeat(28, 1fr)',
          gap: '1px'
        }}>
          {heatMapData.map((cell, index) => (
            <div
              key={index}
              style={{
                backgroundColor: getColor(cell.intensity),
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                borderRadius: '2px'
              }}
              title={`Zone: ${cell.zone || 'N/A'} - Density: ${cell.value}% - Intensity: ${(cell.intensity * 100).toFixed(1)}%`}
              onMouseEnter={(e) => {
                e.target.style.transform = 'scale(1.1)';
                e.target.style.zIndex = '10';
                e.target.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'scale(1)';
                e.target.style.zIndex = '1';
                e.target.style.boxShadow = 'none';
              }}
            />
          ))}
        </div>
      </div>
      
      <div style={{ 
        marginTop: '1rem', 
        padding: '1rem', 
        backgroundColor: '#f3f4f6', 
        borderRadius: '8px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
            Last Updated: {new Date().toLocaleTimeString()}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: '#dc2626' }}>
              {Math.floor(statistics.totalOccupancy / 20) || 150}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>Peak Density</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: '#059669' }}>
              {statistics.averageUsage || 70}%
            </div>
            <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>Capacity</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeatMap;
