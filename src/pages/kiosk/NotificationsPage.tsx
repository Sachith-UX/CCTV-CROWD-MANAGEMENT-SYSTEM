import React, { useEffect, useState } from 'react';

interface Notification {
  time: string;
  message: string;
}

interface NotificationsPageTailwindProps {}

const API_URL = 'http://localhost:3000/api/events/:id/status'; // API endpoint for fetching notifications

const NotificationsPageTailwind: React.FC<NotificationsPageTailwindProps> = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]); // State to hold fetched notifications
  const [loading, setLoading] = useState<boolean>(true); // State to track loading status

  useEffect(() => {
    let intervalId: NodeJS.Timeout; // to store the polling interval

    // Function to fetch notifications from backend
    const fetchNotifications = async () => {
      try {
        setLoading(true); // show loading message
        const response = await fetch(API_URL); // call API
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data: Notification[] = await response.json(); // parse response
        setNotifications(data);  // update state
      } catch (error) {
        console.error('Error fetching notifications:', error);
        // Set mock data for demo purposes when API fails
        setNotifications([
          {
            time: new Date().toLocaleTimeString(),
            message: "Welcome to EngEx 2025! Check out the latest innovations from our engineering departments."
          },
          {
            time: new Date(Date.now() - 5 * 60000).toLocaleTimeString(),
            message: "New project showcase starting in Hall A - AI and Machine Learning innovations."
          },
          {
            time: new Date(Date.now() - 15 * 60000).toLocaleTimeString(),
            message: "Reminder: Industry networking session begins at 3:00 PM in the main auditorium."
          },
          {
            time: new Date(Date.now() - 30 * 60000).toLocaleTimeString(),
            message: "Student competition results will be announced at 4:30 PM today."
          }
        ]);
      } finally {
        setLoading(false); // hide loading after fetch (success/failure)
      }
    };

    // Fetch once immediately
    fetchNotifications();
    intervalId = setInterval(fetchNotifications, 5000); // Set up polling every 5 seconds
    return () => clearInterval(intervalId); // Cleanup: clear interval when component unmounts
  }, []); // [] → run only once on mount

  return (
    <div className="backdrop-blur-md bg-white/20 rounded-2xl shadow-[0_4px_30px_rgba(0,0,0,0.1)] p-8 my-8 mx-auto max-w-4xl">
      <div className="flex items-center gap-4 mb-8">
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-3 rounded-full shadow-lg">
          <span className="text-3xl">🔔</span>
        </div>
        <div>
          <h2 className="text-4xl font-bold text-white drop-shadow-lg mb-2">
            Live Notifications
          </h2>
          <p className="text-white/80 text-lg">
            Real-time updates and announcements
          </p>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-400 border-t-transparent mx-auto mb-4"></div>
          <p className="text-white/80 text-xl">Loading notifications...</p>
        </div>
      ) : notifications.length === 0 ? (
        <div className="text-center py-12">
          <div className="bg-white/10 rounded-2xl p-8 backdrop-blur-sm border border-white/20">
            <span className="text-6xl mb-4 block">📭</span>
            <p className="text-white/80 text-xl">No notifications available at the moment.</p>
            <p className="text-white/60 text-sm mt-2">Check back later for updates!</p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {notifications.map((note, idx) => (
            <div 
              key={idx} 
              className="bg-white/40 backdrop-blur-sm rounded-xl p-6 shadow-[0_2px_8px_rgba(0,0,0,0.05)] border border-white/30 transition-all duration-300 hover:transform hover:-translate-y-1 hover:shadow-[0_4px_15px_rgba(0,0,0,0.1)] animate-slideIn"
              style={{ animationDelay: `${idx * 0.1}s` }}
            >
              <div className="flex items-start gap-4">
                <div className="bg-blue-500/20 p-2 rounded-full flex-shrink-0 mt-1">
                  <span className="text-blue-600 text-sm">🔔</span>
                </div>
                <div className="flex-1">
                  <span className="text-gray-600 text-sm font-medium block mb-2">
                    {note.time}
                  </span>
                  <p className="text-gray-800 text-lg leading-relaxed m-0">
                    {note.message}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Auto-refresh indicator */}
      <div className="mt-6 flex items-center justify-center gap-2 text-white/60 text-sm">
        <div className="animate-pulse w-2 h-2 bg-green-400 rounded-full"></div>
        <span>Auto-refreshing every 5 seconds</span>
      </div>

      {/* Custom animations */}
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateY(20px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
          .animate-slideIn {
            animation: slideIn 0.5s ease-out forwards;
          }
        `
      }} />
    </div>
  );
}

export default NotificationsPageTailwind;
