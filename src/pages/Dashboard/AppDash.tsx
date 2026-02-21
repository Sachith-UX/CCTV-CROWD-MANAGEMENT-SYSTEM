import React, { useEffect, useState } from "react";
import OrganizerDashBoard from "./OrganizerDashBoard";
import LoginPage from "./LoginPage";
import RegisterPage from "./RegisterPage";
import OverviewPage from "./OverviewPage";
import HeatmapsPage from "./HeatmapsPage";
import FeedbackPage from "./FeedbackPage";
import ExportPage from "./ExportPage";
import BuildingsPage from "./BuildingsPage";
import EventsPage from "./EventsPage";
import { Routes, Route, Navigate } from "react-router-dom";

function AppDashboard() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    // Check if user is already logged in
    const storedUser = localStorage.getItem("authUser");
    if (storedUser) {
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem("authUser");
    setIsAuthenticated(false);
  };

  return (
    <Routes>
      <Route path="login" element={<LoginPage onLogin={handleLogin} />} />
      <Route path="register" element={<RegisterPage />} />
      
      {/* Protected routes with nested structure */}
      <Route
        path="/"
        element={
          isAuthenticated ? (
            <OrganizerDashBoard onLogout={handleLogout} />
          ) : (
            <Navigate to="login" replace />
          )
        }
      >
        <Route index element={<OverviewPage />} />
        <Route path="overview" element={<OverviewPage />} />
        <Route path="heatmaps" element={<HeatmapsPage />} />
        <Route path="feedback" element={<FeedbackPage />} />
        <Route path="export" element={<ExportPage />} />
        <Route path="buildings" element={<BuildingsPage />} />
        <Route path="events" element={<EventsPage />} />
      </Route>
      
      <Route path="*" element={<Navigate to={isAuthenticated ? "" : "login"} replace />} />
    </Routes>
  );
}

export default AppDashboard;
