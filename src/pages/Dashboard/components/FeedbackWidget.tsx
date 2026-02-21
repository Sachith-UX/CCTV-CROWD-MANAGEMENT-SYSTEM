import React, { useState } from "react";
import { Star, MessageSquare, ThumbsUp, AlertCircle } from "lucide-react";

interface Feedback {
  rating: number;
  comment: string;
  time: string;
  sentiment: "positive" | "neutral" | "negative";
  zone: string;
  Building: string;
  
}

const FeedbackWidget: React.FC = () => {
  const [sentimentFilter, setSentimentFilter] = useState("all");
  const [zoneFilter, setZoneFilter] = useState("all");
  const [BuildingFilter, setBuildingFilter] = useState("all");

  // Replace these mock stats with API/DB data later
  const feedbackStats = [
    { label: "Overall Rating", value: "4.7/5", icon: Star, color: "text-yellow-500" },
    { label: "Total Reviews", value: "1,247", icon: MessageSquare, color: "text-blue-500" },
    { label: "Satisfaction Rate", value: "92%", icon: ThumbsUp, color: "text-green-500" },
  ];

  // Replace mock feedback with backend data later
  const feedback: Feedback[] = [
    { rating: 5, comment: "Excellent event organization!", time: "2h ago", sentiment: "positive", zone: "Zone A", Building: "Building1"},
    { rating: 4, comment: "Good content, but crowded.", time: "3h ago", sentiment: "neutral", zone: "Zone B", Building: "Building2"},
    { rating: 3, comment: "WiFi issues in the hall.", time: "5h ago", sentiment: "negative", zone: "Zone A", Building: "Building1"},
    { rating: 5, comment: "Loved the keynote!", time: "1h ago", sentiment: "positive", zone: "Zone B", Building: "Building2"},
  ];

  const filteredFeedback = feedback.filter(f => {
    const matchSentiment = sentimentFilter === "all" || f.sentiment === sentimentFilter;
    const matchZone = zoneFilter === "all" || f.zone === zoneFilter;
    const matchBuilding = BuildingFilter === "all" || f.Building === BuildingFilter;
    return matchSentiment && matchZone && matchBuilding;
  });

  const zones = ["all", ...Array.from(new Set(feedback.map(f => f.zone)))];
  const Buildings = ["all", ...Array.from(new Set(feedback.map(f => f.Building)))];

  const getSentimentColor = (sentiment: string) => {
    const colors = {
      positive: "border-l-green-500 bg-green-50",
      neutral: "border-l-yellow-500 bg-yellow-50",
      negative: "border-l-red-500 bg-red-50",
    };
    return colors[sentiment as keyof typeof colors];
  };

  return (
    <div className="space-y-6">
      {/* Feedback Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {feedbackStats.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={i} className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center space-x-3">
                <Icon size={24} className={stat.color} />
                <div>
                  <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                  <p className="text-sm text-gray-600">{stat.label}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap justify-end gap-4 mb-4">
        <select value={sentimentFilter} onChange={(e) => setSentimentFilter(e.target.value)} className="border rounded px-3 py-2">
          <option value="all">All Sentiments</option>
          <option value="positive">Positive</option>
          <option value="neutral">Neutral</option>
          <option value="negative">Negative</option>
        </select>

        <select value={zoneFilter} onChange={(e) => setZoneFilter(e.target.value)} className="border rounded px-3 py-2">
          {zones.map((z, i) => (
            <option key={i} value={z}>{z === "all" ? "All Zones" : z}</option>
          ))}
        </select>

        <select value={BuildingFilter} onChange={(e) => setBuildingFilter(e.target.value)} className="border rounded px-3 py-2">
          {Buildings.map((s, i) => (
            <option key={i} value={s}>{s === "all" ? "All Buildings" : s}</option>
          ))}
        </select>

        
      </div>

      {/* Recent Feedback */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Recent Feedback</h3>
          <button className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            View All
          </button>
        </div>

        <div className="space-y-4">
          {filteredFeedback.length === 0 ? (
            <p className="text-gray-500 text-sm">No feedback matches the selected filters.</p>
          ) : (
            filteredFeedback.map((fb, i) => (
              <div key={i} className={`border-l-4 p-4 rounded-lg ${getSentimentColor(fb.sentiment)}`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="flex space-x-1">
                        {[...Array(5)].map((_, idx) => (
                          <Star
                            key={idx}
                            size={16}
                            className={idx < fb.rating ? "text-yellow-400 fill-current" : "text-gray-300"}
                          />
                        ))}
                      </div>
                      <span className="text-sm text-gray-500">{fb.time}</span>
                    </div>
                    <p className="text-gray-700">{fb.comment}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      📍 Zone: {fb.zone} | Building: {fb.Building}
                    </p>
                  </div>
                  {fb.sentiment === "negative" && <AlertCircle size={20} className="text-red-500 ml-2 flex-shrink-0" />}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default FeedbackWidget;

