import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Home, Users, Calendar, Info, Map } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
}

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const location = useLocation();

  const navItems: NavItem[] = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/crowd-management', label: 'Crowd Management', icon: Users },
    { path: '/dashboard', label: 'Dashboard', icon: Users },
    { path: '/map', label: 'Map', icon: Map },
    { path: '/events', label: 'Events', icon: Calendar },
    { path: '/information', label: 'Information', icon: Info },
  ];

  const toggleMenu = (): void => {
    setIsOpen(!isOpen);
  };

  return (
    <nav className="fixed top-0 left-0 right-0 bg-gray-800 z-[1000] px-5 h-[70px] flex items-center justify-between shadow-md">
      <div className="flex items-center">
        <Link 
          to="/" 
          className="text-white no-underline text-2xl font-bold flex items-center"
        >
          <img 
            src="./engex.png" 
            alt="EngEx 2025" 
            className="h-10 mr-2" 
          />
          EngEx 2025
        </Link>
      </div>

      {/* Desktop Menu */}
      <div className="hidden md:flex items-center gap-8">
        {navItems.map((item) => {
          const IconComponent = item.icon;
          const isActive = location.pathname === item.path;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`text-gray-300 no-underline flex items-center gap-2 px-4 py-2 rounded-lg bg-transparent transition-all duration-200 text-sm font-medium hover:text-gray-100 hover:bg-gray-600 hover:bg-opacity-50 ${
                isActive ? 'text-blue-400 bg-blue-500 bg-opacity-10' : ''
              }`}
            >
              <IconComponent size={18} />
              {item.label}
            </Link>
          );
        })}
      </div>

      {/* Mobile Menu Button */}
      <button
        onClick={toggleMenu}
        className="md:hidden bg-transparent border-none text-white cursor-pointer p-2"
      >
        {isOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="absolute top-full left-0 right-0 bg-gray-800 p-4 shadow-md border-t border-gray-700 md:hidden">
          {navItems.map((item) => {
            const IconComponent = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setIsOpen(false)}
                className={`text-gray-300 no-underline flex items-center gap-3 p-3 rounded-lg bg-transparent mb-2 text-base font-medium transition-all duration-200 hover:text-gray-100 hover:bg-gray-600 hover:bg-opacity-50 ${
                  isActive ? 'text-blue-400 bg-blue-500 bg-opacity-10' : ''
                }`}
              >
                <IconComponent size={20} />
                {item.label}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
