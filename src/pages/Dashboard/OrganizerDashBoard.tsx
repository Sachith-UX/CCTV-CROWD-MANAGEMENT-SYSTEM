import React, { useState } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";

interface OrganizerDashBoardProps {
  onLogout: () => void;
}

function OrganizerDashBoard({ onLogout }: OrganizerDashBoardProps) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <Header
          eventInfo={{
            title: "EngEX 2025",
            date: "30 Aug 2025",
            location: "University of Peradeniya",
          }}
          userInfo={{
            name: "Organizer",
            role: "Admin",
          }}
          onLogout={onLogout}
        />

        {/* Nested routes render here */}
        <main className="flex-1 p-6 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default OrganizerDashBoard;

