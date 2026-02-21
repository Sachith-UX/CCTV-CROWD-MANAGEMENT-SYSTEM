import { useState, useEffect } from 'react'
// @ts-ignore - JavaScript module without TypeScript declarations
import { getAllEvents, getEventsWithinHour, formatEventForDisplay } from './utils/eventService'

interface Event {
  id?: any;
  title: any;
  venue: any;
  time: string;
  duration: string;
  status: string;
  timeInfo?: string;
  type?: string;
  startDateTime?: Date;
  endDateTime?: Date;
  isToday?: boolean;
}

interface SchedulePageTailwindProps {}

const SchedulePageTailwind: React.FC<SchedulePageTailwindProps> = () => {
  // State management
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [currentEvents, setCurrentEvents] = useState<Event[]>([]) // Events within an hour
  const [allEvents, setAllEvents] = useState<Event[]>([]) // All events for search
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [searchResults, setSearchResults] = useState<number>(0)
  const [activeTab, setActiveTab] = useState<'happening' | 'all'>('happening')
  
  // Fetch events from database
  useEffect(() => {
    const fetchEvents = async () => {
      try {
        setLoading(true)
        setError(null)
        
        // Fetch both current events and all events
        const [currentResult, allResult] = await Promise.all([
          getEventsWithinHour(),
          getAllEvents()
        ])
        
        if (currentResult.success && allResult.success && currentResult.data && allResult.data) {
          // Format events for display
          const formattedCurrentEvents = currentResult.data.map(formatEventForDisplay)
          const formattedAllEvents = allResult.data.map(formatEventForDisplay)
          
          setCurrentEvents(formattedCurrentEvents)
          setAllEvents(formattedAllEvents)
        } else {
          setError(currentResult.error || allResult.error || 'Failed to fetch events')
        }
      } catch (err) {
        setError('An unexpected error occurred')
        console.error('Error fetching events:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchEvents()
    
    // Set up interval to refresh events every minute
    const interval = setInterval(fetchEvents, 60000) // Refresh every minute
    
    return () => clearInterval(interval)
  }, [])

  // Enhanced filter function - searches in all events when there's a query
  const getDisplayEvents = (): Event[] => {
    if (searchQuery.trim()) {
      // Search query exists - search through all events
      const query = searchQuery.toLowerCase().trim()
      
      const filteredEvents = allEvents.filter(event => {
        // Check if query matches status keywords
        const statusMatch = 
          (query === 'ongoing' && event.status === 'ongoing') ||
          (query === 'upcoming' && event.status === 'upcoming') ||
          (query === 'completed' && event.status === 'completed')
        
        // Check if query is in title
        const titleMatch = event.title.toLowerCase().includes(query)
        
        // Check if query is in venue/location
        const locationMatch = event.venue.toLowerCase().includes(query)
        
        // Check if query is in time or duration (secondary matches)
        const timeMatch = event.time.toLowerCase().includes(query)
        const durationMatch = event.duration.toLowerCase().includes(query)
        const typeMatch = event.type && event.type.toLowerCase().includes(query)
        
        // Return true if any of the fields match
        return statusMatch || titleMatch || locationMatch || timeMatch || durationMatch || typeMatch
      })
      
      return filteredEvents
    } else {
      // No search query - show events based on active tab
      return activeTab === 'happening' ? currentEvents : allEvents
    }
  }

  const displayEvents = getDisplayEvents()
  
  // Update search results count after filtering (outside of render function)
  useEffect(() => {
    if (searchQuery.trim()) {
      setSearchResults(displayEvents.length)
    }
  }, [searchQuery, displayEvents.length])

  // Function to get status color
  const getStatusColor = (status: string): string => {
    switch(status) {
      case 'ongoing': return "#10b981" // Green
      case 'upcoming': return "#3b82f6" // Blue
      case 'completed': return "#6b7280" // Gray
      default: return "#6b7280" // Default Gray
    }
  }

  // Function to get status text
  const getStatusText = (status: string): string => {
    switch(status) {
      case 'ongoing': return "ONGOING"
      case 'upcoming': return "UPCOMING"
      case 'completed': return "COMPLETED"
      default: return "UNKNOWN"
    }
  }

  // Function to highlight search terms in text
  const highlightSearchTerm = (text: string, searchTerm: string): React.ReactNode => {
    if (!searchTerm.trim()) return text
    
    const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    const parts = text.split(regex)
    
    return parts.map((part, index) => 
      regex.test(part) ? 
        <span key={index} className="bg-yellow-300 text-black px-1 rounded font-semibold">{part}</span> : 
        part
    )
  }

  // Clear search function
  const clearSearch = () => {
    setSearchQuery('')
  }

  return (
    <div className="w-full h-full p-0 box-border overflow-visible animate-fadeIn">
      <div className="max-w-7xl mx-auto p-8 h-auto w-full">
        
        {/* Page Header Section */}
        <div className="mb-8 text-center">
          <h1 className="text-5xl text-white mb-8 font-semibold drop-shadow-[0_2px_8px_rgba(0,0,0,0.3)]">
            {searchQuery ? 'Event Search Results' : 'Events Schedule'}
          </h1>
          
          {/* Tab Navigation */}
          <div className="flex justify-center gap-4 mb-4">
            <button 
              className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                activeTab === 'happening' 
                  ? 'bg-blue-500/30 text-white border-2 border-blue-400/60 shadow-[0_0_20px_rgba(59,130,246,0.4)]' 
                  : 'bg-white/10 text-white/80 border-2 border-white/20 hover:bg-white/20 hover:text-white'
              }`}
              onClick={() => setActiveTab('happening')}
            >
              Events Happening
            </button>
            <button 
              className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                activeTab === 'all' 
                  ? 'bg-blue-500/30 text-white border-2 border-blue-400/60 shadow-[0_0_20px_rgba(59,130,246,0.4)]' 
                  : 'bg-white/10 text-white/80 border-2 border-white/20 hover:bg-white/20 hover:text-white'
              }`}
              onClick={() => setActiveTab('all')}
            >
              All Events
            </button>
          </div>
          
          {/* Tab Description */}
          <p className="text-white/80 text-lg mb-6">
            {activeTab === 'happening' 
              ? 'Events happening now or starting within the next hour'
              : 'Complete schedule of all upcoming events'}
          </p>
          
          {/* Enhanced Search Input */}
          <div className="max-w-2xl mx-auto">
            <div className="relative">
              <input
                type="text"
                placeholder="Search by status, title, or location..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full p-4 text-lg bg-white/90 border-none rounded-3xl text-gray-700 shadow-[0_4px_15px_rgba(0,0,0,0.1)] transition-all duration-300 focus:outline-none focus:shadow-[0_4px_20px_rgba(59,130,246,0.3)] focus:bg-white placeholder:text-gray-600 placeholder:opacity-80"
              />
              {searchQuery && (
                <button 
                  onClick={clearSearch}
                  className="absolute right-4 top-1/2 transform -translate-y-1/2 w-8 h-8 bg-gray-400/20 hover:bg-gray-400/40 rounded-full flex items-center justify-center text-gray-600 hover:text-gray-800 transition-all duration-200"
                  title="Clear search"
                >
                  ✕
                </button>
              )}
            </div>
            
            {/* Search Results Info */}
            {searchQuery && (
              <div className="mt-3 text-center">
                {searchResults > 0 ? (
                  <span className="text-green-400 font-medium">
                    Found {searchResults} event{searchResults !== 1 ? 's' : ''} matching "{searchQuery}"
                  </span>
                ) : (
                  <span className="text-red-400 font-medium">
                    No events found matching "{searchQuery}"
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Events Display Section */}
        <div className="flex flex-col gap-4 mt-8">
          {loading ? (
            <div className="text-center p-12">
              <p className="text-white text-xl">Loading events...</p>
            </div>
          ) : error ? (
            <div className="text-center p-12">
              <p className="text-red-400 text-xl mb-4">Error: {error}</p>
              <button 
                onClick={() => window.location.reload()} 
                className="px-6 py-3 bg-blue-500/30 text-white rounded-xl border border-blue-400/60 hover:bg-blue-500/50 transition-all duration-300"
              >
                Retry
              </button>
            </div>
          ) : displayEvents.length > 0 ? (
            // Map through display events and show each one
            displayEvents.map((event, index) => (
              <div 
                key={event.id || index} 
                className="relative bg-white/10 backdrop-blur-xl rounded-2xl border border-white/20 p-6 shadow-[0_8px_32px_rgba(0,0,0,0.3)] transition-all duration-300 hover:transform hover:-translate-y-1 hover:shadow-[0_12px_40px_rgba(0,0,0,0.4)] animate-slideIn"
                style={{animationDelay: `${index * 0.1}s`}}
              >
                {/* Status Badge */}
                <div 
                  className="absolute top-4 right-4 px-3 py-1 rounded-full text-white text-sm font-bold shadow-lg"
                  style={{ backgroundColor: getStatusColor(event.status) }}
                >
                  <span>
                    {getStatusText(event.status)}
                  </span>
                </div>
                
                {/* Event Information */}
                <div className="pr-24">
                  {/* Main event info */}
                  <div className="mb-4">
                    <h3 className="text-2xl font-bold text-white mb-2">
                      {highlightSearchTerm(event.title, searchQuery)}
                    </h3>
                    <p className="text-blue-300 text-lg">
                      Duration: {highlightSearchTerm(event.duration, searchQuery)}
                    </p>
                  </div>
                  
                  {/* Event details */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                      <span className="text-white/70 text-sm font-medium block mb-1">Location</span>
                      <span className="text-white text-lg">
                        {highlightSearchTerm(event.venue, searchQuery)}
                      </span>
                    </div>
                    
                    <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                      <span className="text-white/70 text-sm font-medium block mb-1">Start Time</span>
                      <span className="text-white text-lg">
                        {highlightSearchTerm(event.time, searchQuery)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            // Show appropriate message based on search state
            <div className="text-center p-12">
              {searchQuery ? (
                <div className="space-y-4">
                  <p className="text-white/70 text-xl">No events found matching "{searchQuery}"</p>
                  <button 
                    onClick={clearSearch} 
                    className="px-6 py-3 bg-blue-500/30 text-white rounded-xl border border-blue-400/60 hover:bg-blue-500/50 transition-all duration-300"
                  >
                    Show Current Events
                  </button>
                </div>
              ) : activeTab === 'happening' ? (
                <div className="space-y-4">
                  <p className="text-white/70 text-xl">No events are currently happening or starting within the next hour.</p>
                  <button 
                    onClick={() => setActiveTab('all')} 
                    className="px-6 py-3 bg-blue-500/30 text-white rounded-xl border border-blue-400/60 hover:bg-blue-500/50 transition-all duration-300"
                  >
                    View All Events
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  <p className="text-white/70 text-xl">No events are currently scheduled.</p>
                  <p className="text-white/50">Check back later for updates!</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Custom animations */}
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
          }
          .animate-fadeIn { animation: fadeIn 0.3s ease-out; }
          .animate-slideIn { animation: slideIn 0.5s ease-out forwards; }
        `
      }} />
    </div>
  )
}

export default SchedulePageTailwind
