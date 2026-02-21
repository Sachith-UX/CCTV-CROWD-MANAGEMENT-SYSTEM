import { useState, useEffect, useRef, useCallback } from 'react'
import HomePage from './HomePage'
import AboutPage from './AboutPage'
import SchedulePage from './SchedulePage'
import NotificationsPage from './NotificationsPage'
import MapPage from './MapPage'
import HeatMapPage from './HeatMapPage'
import ContactPage from './ContactPage'
import IntroVideo from './IntroVideo'
import Navigation from './Navigation'
import Footer from './Footer'

interface AppKioskProps {}

const AppKiosk: React.FC<AppKioskProps> = () => {
  const [currentPage, setCurrentPage] = useState<number>(0)
  const [showIntroVideo, setShowIntroVideo] = useState<boolean>(true)
  const inactivityTimerRef = useRef<NodeJS.Timeout | null>(null)
  const pages = [HomePage, AboutPage, SchedulePage, NotificationsPage, MapPage, HeatMapPage, ContactPage]

  const handleUserActivity = useCallback(() => {
    if (inactivityTimerRef.current) {
      clearTimeout(inactivityTimerRef.current)
    }
    if (!showIntroVideo) {
      inactivityTimerRef.current = setTimeout(() => {
        setShowIntroVideo(true)
        setCurrentPage(0)
      }, 60000)
    }
  }, [showIntroVideo])

  const handleIntroVideoClick = () => {
    setShowIntroVideo(false)
    setCurrentPage(0)
    handleUserActivity()
  }

  const handlePageClick = (pageIndex: number) => {
    setCurrentPage(pageIndex)
    handleUserActivity()
  }

  useEffect(() => {
    const events = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'] as const
    const addEventListeners = () => {
      events.forEach(event => {
        document.addEventListener(event, handleUserActivity, true)
      })
    }
    const removeEventListeners = () => {
      events.forEach(event => {
        document.removeEventListener(event, handleUserActivity, true)
      })
    }
    if (!showIntroVideo) {
      addEventListeners()
      handleUserActivity()
    } else {
      removeEventListeners()
      if (inactivityTimerRef.current) {
        clearTimeout(inactivityTimerRef.current)
      }
    }
    return () => {
      removeEventListeners()
      if (inactivityTimerRef.current) {
        clearTimeout(inactivityTimerRef.current)
      }
    }
  }, [showIntroVideo, handleUserActivity])

  const CurrentComponent = pages[currentPage]

  if (showIntroVideo) {
    return <IntroVideo onVideoClick={handleIntroVideoClick} />
  }

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-slate-800 via-slate-700 to-slate-800 bg-cover bg-center bg-no-repeat bg-fixed relative flex flex-col overflow-hidden text-base">
      {/* Overlay gradients and patterns */}
      <div 
        className="absolute inset-0 z-0 pointer-events-none"
        style={{
          background: `radial-gradient(circle at 20% 80%, rgba(56,189,248,0.1) 0%, transparent 50%),
                      radial-gradient(circle at 80% 20%, rgba(59,130,246,0.08) 0%, transparent 50%),
                      radial-gradient(circle at 40% 40%, rgba(6,182,212,0.12) 0%, transparent 50%),
                      linear-gradient(180deg, rgba(56,189,248,0.05) 0%, transparent 100%)`
        }}
      />
      {/* Navigation */}
      <Navigation 
        currentPage={currentPage}
        onPageClick={handlePageClick}
        pages={pages}
      />
      {/* Main Content */}
      <div className="flex-1 flex flex-col ml-[220px] mr-5 mt-2 mb-2 h-[calc(100vh-60px)]">
        <div className="flex-1 p-8 overflow-y-auto overflow-x-hidden flex flex-col bg-slate-700/30 backdrop-blur-xl rounded-3xl mb-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.1),0_20px_40px_rgba(0,0,0,0.15)] scroll-smooth scrollbar-thin scrollbar-thumb-sky-400/60 scrollbar-track-slate-800/30">
          <CurrentComponent />
        </div>
        <Footer />
      </div>
    </div>
  )
}

export default AppKiosk;
