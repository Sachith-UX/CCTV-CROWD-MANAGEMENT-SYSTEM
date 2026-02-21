import { supabase } from '../supabaseClient'

// Get all events
export async function getAllEvents() {
  try {
    const { data, error } = await supabase
      .from('events')
      .select('*')
      .order('date', { ascending: true })
      .order('starttime', { ascending: true })
    
    if (error) throw error
    
    return { success: true, data }
  } catch (error) {
    console.error('Error fetching events:', error)
    return { success: false, error: error.message }
  }
}

// Get events for today
export async function getTodayEvents() {
  try {
    const today = new Date().toISOString().split('T')[0]
    
    const { data, error } = await supabase
      .from('events')
      .select('*')
      .eq('date', today)
      .order('starttime', { ascending: true })
    
    if (error) throw error
    
    return { success: true, data }
  } catch (error) {
    console.error('Error fetching today events:', error)
    return { success: false, error: error.message }
  }
}

// Get upcoming events
export async function getUpcomingEvents(limit = 10) {
  try {
    const today = new Date().toISOString().split('T')[0]
    
    const { data, error } = await supabase
      .from('events')
      .select('*')
      .gte('date', today)
      .order('date', { ascending: true })
      .order('starttime', { ascending: true })
      .limit(limit)
    
    if (error) throw error
    
    return { success: true, data }
  } catch (error) {
    console.error('Error fetching upcoming events:', error)
    return { success: false, error: error.message }
  }
}

// Format event data for display in SchedulePage
export function formatEventForDisplay(event) {
  const now = new Date()
  const eventDate = new Date(event.date)
  const today = new Date()
  today.setHours(0, 0, 0, 0) // Reset time to start of day
  
  // Create full datetime objects for start and end times
  const eventStartDateTime = new Date(`${event.date}T${event.starttime}`)
  const eventEndDateTime = new Date(`${event.date}T${event.endtime}`)
  
  // Determine event status based on current time
  let status = 'upcoming'
  
  if (now >= eventEndDateTime) {
    // Event has ended
    status = 'completed'
  } else if (now >= eventStartDateTime && now < eventEndDateTime) {
    // Event is currently happening
    status = 'ongoing'
  } else if (now < eventStartDateTime) {
    // Event hasn't started yet
    status = 'upcoming'
  }
  
  // Format time for display (convert from 24hr to 12hr format)
  const formatTime = (timeString) => {
    const [hours, minutes] = timeString.split(':')
    const hour = parseInt(hours)
    const ampm = hour >= 12 ? 'PM' : 'AM'
    const displayHour = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour
    return `${displayHour}:${minutes} ${ampm}`
  }
  
  // Calculate time remaining or elapsed
  const getTimeInfo = () => {
    const diffMs = eventStartDateTime - now
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
    const diffMinutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60))
    
    if (status === 'upcoming' && diffHours >= 0) {
      if (diffHours > 24) {
        const days = Math.floor(diffHours / 24)
        return `Starts in ${days} day${days > 1 ? 's' : ''}`
      } else if (diffHours > 0) {
        return `Starts in ${diffHours}h ${diffMinutes}m`
      } else if (diffMinutes > 0) {
        return `Starts in ${diffMinutes} minutes`
      } else {
        return 'Starting soon'
      }
    } else if (status === 'ongoing') {
      const endDiffMs = eventEndDateTime - now
      const endMinutes = Math.floor(endDiffMs / (1000 * 60))
      if (endMinutes > 60) {
        const endHours = Math.floor(endMinutes / 60)
        return `Ends in ${endHours}h ${endMinutes % 60}m`
      } else {
        return `Ends in ${endMinutes} minutes`
      }
    }
    return ''
  }
  
  return {
    id: event.id,
    title: event.title,
    venue: event.location,
    time: formatTime(event.starttime),
    duration: `${formatTime(event.starttime)} - ${formatTime(event.endtime)}`,
    status: status,
    timeInfo: getTimeInfo(),
    type: 'event',
    // Additional fields for better status display
    startDateTime: eventStartDateTime,
    endDateTime: eventEndDateTime,
    isToday: eventDate.toDateString() === today.toDateString()
  }
}

// Get events that are ongoing or will start within an hour from current time
export async function getEventsWithinHour() {
  try {
    const now = new Date()
    const oneHourFromNow = new Date(now.getTime() + (60 * 60 * 1000)) // Add 1 hour
    
    // Get today's date in YYYY-MM-DD format
    const today = now.toISOString().split('T')[0]
    const tomorrow = new Date(now.getTime() + (24 * 60 * 60 * 1000)).toISOString().split('T')[0]
    
    const { data, error } = await supabase
      .from('events')
      .select('*')
      .or(`date.eq.${today},date.eq.${tomorrow}`) // Include today and tomorrow to handle edge cases
      .order('date', { ascending: true })
      .order('starttime', { ascending: true })
    
    if (error) throw error
    
    // Filter events that are ongoing or start within the next hour
    const filteredEvents = data.filter(event => {
      const eventStartDateTime = new Date(`${event.date}T${event.starttime}`)
      const eventEndDateTime = new Date(`${event.date}T${event.endtime}`)
      
      // Include if:
      // 1. Event is currently ongoing (started but not ended)
      // 2. Event will start within the next hour
      return (
        (now >= eventStartDateTime && now < eventEndDateTime) || // Ongoing
        (eventStartDateTime >= now && eventStartDateTime <= oneHourFromNow) // Starting within hour
      )
    })
    
    return { success: true, data: filteredEvents }
  } catch (error) {
    console.error('Error fetching events within hour:', error)
    return { success: false, error: error.message }
  }
}