import React, { useState } from "react";
import { MapPin, Thermometer, Activity, Clock, Users } from "lucide-react";

interface Zone {
  name: string;
  peak: number;
  dwell: number;
  activity: "High" | "Medium" | "Low";
  color: "red" | "yellow" | "green";
  icon: typeof Users;
}

const HeatmapWidget: React.FC = () => {
  const [timeFilter, setTimeFilter] = useState("24h");
  const [zoneFilter, setZoneFilter] = useState("all");
  const [BuildingFilter, setBuildingFilter] = useState("all");

  const allZones: Zone[] = [
    { name: "Main Hall", peak: 187, dwell: 28, activity: "High", color: "red", icon: Users },
    { name: "Exhibition Area", peak: 134, dwell: 22, activity: "Medium", color: "yellow", icon: Activity },
    { name: "Networking Lounge", peak: 89, dwell: 35, activity: "Low", color: "green", icon: Clock },
    { name: "Computer Engineering Dept", peak: 130, dwell: 20, activity: "Medium", color: "yellow", icon: Activity },
    { name: "Competition Area", peak: 90, dwell: 18, activity: "Low", color: "green", icon: Clock },
  ];

  const filteredZones =
    zoneFilter === "all" ? allZones : allZones.filter((z) => z.name === zoneFilter);

  const colorClasses: Record<string, string> = {
    red: "from-red-500 to-red-600",
    yellow: "from-yellow-500 to-yellow-600",
    green: "from-green-500 to-green-600",
  };

  return (
    <div className="space-y-8">
      {/* 🔽 Filters */}
      <div className="flex flex-wrap justify-end gap-3">
        <select value={timeFilter} onChange={(e) => setTimeFilter(e.target.value)} className="border px-3 py-2 rounded">
          <option value="1h">Last 1 Hour</option>
          <option value="3h">Last 3 Hours</option>
          <option value="5h">Last 5 Hours</option>
          <option value="12h">Last 12 Hours</option>
          <option value="24h">Last 24 Hours</option>
        </select>
        <select value={zoneFilter} onChange={(e) => setZoneFilter(e.target.value)} className="border px-3 py-2 rounded">
          <option value="all">All Zones</option>
          {allZones.map((z, i) => (
            <option key={i} value={z.name}>
              {z.name}
            </option>
          ))}
        </select>
        <select value={BuildingFilter} onChange={(e) => setBuildingFilter(e.target.value)} className="border px-3 py-2 rounded">
          <option value="all">All Buildings</option>
          <option value="Building1">Building 1</option>
          <option value="Building2">Building 2</option>
        </select>
        
      </div>

      {/* 📍 Heatmap Visualization */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-white/20 p-8 shadow-xl hover:shadow-2xl transition-all duration-300">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-gradient-to-br from-red-500 to-orange-500 rounded-xl text-white shadow-lg">
              <Thermometer size={24} />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-900">Venue Heatmap</h3>
              <p className="text-gray-600">Real-time crowd density visualization</p>
            </div>
          </div>
          <div className="px-4 py-2 bg-blue-50 text-blue-700 rounded-full text-sm font-medium">
            Live Data
          </div>
        </div>

        <div className="relative h-96 bg-gradient-to-br from-blue-50 via-yellow-50 to-red-50 rounded-2xl flex items-center justify-center border border-white/50 overflow-hidden">
          <div className="absolute inset-0 opacity-10">
            <div
              className="w-full h-full"
              style={{
                backgroundImage: `radial-gradient(circle at 1px 1px, rgba(59, 130, 246, 0.3) 1px, transparent 0)`,
                backgroundSize: "20px 20px",
              }}
            ></div>
          </div>
          <div className="text-center relative z-10">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
              <MapPin size={32} className="text-blue-600" />
            </div>
            <p className="text-gray-700 font-medium text-lg">
              Interactive heatmap visualization
            </p>
            <p className="text-gray-500 mt-2">
              Showing {zoneFilter} [{timeFilter}]
            </p>
          </div>

          {/* Animated activity dots */}
          <div className="absolute top-6 left-6 w-4 h-4 bg-red-500 rounded-full animate-pulse"></div>
          <div className="absolute top-20 right-16 w-3 h-3 bg-yellow-500 rounded-full animate-pulse-slow"></div>
          <div className="absolute bottom-16 left-20 w-5 h-5 bg-orange-500 rounded-full animate-pulse"></div>
          <div className="absolute bottom-8 right-8 w-2 h-2 bg-blue-500 rounded-full animate-pulse-slow"></div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center space-x-8 mt-6">
          <div className="flex items-center space-x-3 px-4 py-2 bg-blue-50 rounded-full">
            <div className="w-4 h-4 bg-blue-400 rounded-full"></div>
            <span className="text-sm font-medium text-blue-700">Low Activity</span>
          </div>
          <div className="flex items-center space-x-3 px-4 py-2 bg-yellow-50 rounded-full">
            <div className="w-4 h-4 bg-yellow-400 rounded-full"></div>
            <span className="text-sm font-medium text-yellow-700">Medium Activity</span>
          </div>
          <div className="flex items-center space-x-3 px-4 py-2 bg-red-50 rounded-full">
            <div className="w-4 h-4 bg-red-400 rounded-full"></div>
            <span className="text-sm font-medium text-red-700">High Activity</span>
          </div>
        </div>
      </div>

      {/* 📊 Zone Analytics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {filteredZones.map((zone, i) => (
          <div
            key={i}
            className="group bg-white/80 backdrop-blur-sm rounded-2xl border border-white/20 p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
          >
            <div className="flex items-center space-x-3 mb-4">
              <div className={`p-3 rounded-xl text-white shadow-lg ${colorClasses[zone.color]}`}>
                <zone.icon size={20} />
              </div>
              <div>
                <h4 className="font-bold text-gray-900 text-lg">{zone.name}</h4>
                <div
                  className={`px-2 py-1 rounded-full text-xs font-medium w-fit ${
                    zone.activity === "High"
                      ? "bg-red-100 text-red-700"
                      : zone.activity === "Medium"
                      ? "bg-yellow-100 text-yellow-700"
                      : "bg-green-100 text-green-700"
                  }`}
                >
                  {zone.activity} Activity
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-50/50 rounded-xl">
                <span className="text-sm text-gray-600 flex items-center space-x-2">
                  <Users size={14} />
                  <span>Peak Occupancy:</span>
                </span>
                <span className="text-sm font-bold text-gray-900">{zone.peak}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50/50 rounded-xl">
                <span className="text-sm text-gray-600 flex items-center space-x-2">
                  <Clock size={14} />
                  <span>Avg. Dwell Time:</span>
                </span>
                <span className="text-sm font-bold text-gray-900">{zone.dwell} min</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HeatmapWidget;

