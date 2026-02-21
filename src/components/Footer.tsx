import React from 'react';
import { MapPin, Phone, Mail } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer style={{
      backgroundColor: '#1f2937',
      color: '#d1d5db',
      padding: '3rem 0 1rem',
      marginTop: 'auto'
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '0 1rem'
      }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '2rem',
          marginBottom: '2rem'
        }}>
          <div>
            <h3 style={{ color: '#ffffff', marginBottom: '1rem', fontSize: '1.25rem' }}>
              Event Management System
            </h3>
            <p style={{ lineHeight: '1.6', color: '#9ca3af' }}>
              Advanced crowd management and event scheduling platform for efficient 
              event organization and real-time monitoring.
            </p>
          </div>
          
          <div>
            <h4 style={{ color: '#ffffff', marginBottom: '1rem' }}>Quick Links</h4>
            <ul style={{ listStyle: 'none', lineHeight: '2' }}>
              <li><a href="/" style={{ color: '#9ca3af', textDecoration: 'none' }}>Dashboard</a></li>
              <li><a href="/crowd-management" style={{ color: '#9ca3af', textDecoration: 'none' }}>Crowd Management</a></li>
              <li><a href="/events" style={{ color: '#9ca3af', textDecoration: 'none' }}>Event Schedule</a></li>
              <li><a href="/information" style={{ color: '#9ca3af', textDecoration: 'none' }}>Information</a></li>
            </ul>
          </div>
          
          <div>
            <h4 style={{ color: '#ffffff', marginBottom: '1rem' }}>Contact Info</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <MapPin size={16} />
                <span style={{ color: '#9ca3af' }}>123 Event Street, City, State 12345</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Phone size={16} />
                <span style={{ color: '#9ca3af' }}>+1 (555) 123-4567</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Mail size={16} />
                <span style={{ color: '#9ca3af' }}>contact@e21231@eng.pdn.ac.lk</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              </div>
            </div>
          </div>
          
          <div>
            <h4 style={{ color: '#ffffff', marginBottom: '1rem' }}>Emergency Contacts</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ color: '#ef4444' }}>
                <strong>Emergency: 119</strong>
              </div>
              <div style={{ color: '#f59e0b' }}>
                <strong>Security: +94 (555) 999-0000</strong>
              </div>
              <div style={{ color: '#10b981' }}>
                <strong>Medical: +94 (555) 888-0000</strong>
              </div>
            </div>
          </div>
        </div>
        
        <div style={{
          borderTop: '1px solid #374151',
          paddingTop: '1rem',
          textAlign: 'center',
          color: '#6b7280',
          fontSize: '0.875rem'
        }}>
          <p>&copy; 2025 Event Management System. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
