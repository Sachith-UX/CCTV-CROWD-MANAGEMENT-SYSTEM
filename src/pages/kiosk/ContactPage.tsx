import React, { useState } from 'react';

interface EmergencyContact {
  title: string;
  type: string;
  number: string;
  description: string;
  availability: string;
}

interface MedicalCenter {
  name: string;
  address: string;
  phone: string;
  hours: string;
}

interface EventInfo {
  title: string;
  contact: string;
  email: string;
}

type SectionType = 'emergency' | 'medical' | 'phones' | 'event' | 'important';

interface ContactPageTailwindProps {}

const ContactPageTailwind: React.FC<ContactPageTailwindProps> = () => {
  const [selectedSection, setSelectedSection] = useState<SectionType>('emergency');

  const emergencyContacts: EmergencyContact[] = [
    {
      title: "Emergency Hotline",
      type: "EMERGENCY",
      number: "119",
      description: "National Emergency Services",
      availability: "24/7"
    },
    {
      title: "University Security",
      type: "SECURITY", 
      number: "+94 81 239 4914",
      description: "Campus Security Control Room",
      availability: "24/7"
    },
    {
      title: "Police Emergency",
      type: "POLICE",
      number: "118",
      description: "Sri Lanka Police Emergency",
      availability: "24/7"
    },
    {
      title: "Fire Department",
      type: "FIRE",
      number: "110",
      description: "Fire & Rescue Services",
      availability: "24/7"
    },
    {
      title: "Ambulance Service",
      type: "MEDICAL",
      number: "110",
      description: "Emergency Medical Services",
      availability: "24/7"
    }
  ];

  const medicalCenters: MedicalCenter[] = [
    {
      name: "University Medical Center",
      address: "Faculty of Medicine, University of Peradeniya",
      phone: "+94 81 239 2361",
      hours: "8:00 AM - 5:00 PM"
    },
    {
      name: "Peradeniya Teaching Hospital",
      address: "Peradeniya, Kandy",
      phone: "+94 81 238 8000",
      hours: "24/7"
    },
    {
      name: "Kandy General Hospital",
      address: "Kandy City Center",
      phone: "+94 81 222 2261",
      hours: "24/7"
    }
  ];

  const phoneBooths: string[] = [
    "Engineering Faculty - Main Entrance",
    "Library Building - Ground Floor",
    "Student Center - Near Cafeteria",
    "Administrative Building - Reception",
    "Hostel Complex - Common Area"
  ];

  const eventInfo: EventInfo[] = [
    {
      title: "Event Coordinator",
      contact: "+94 81 239 3000",
      email: "coordinator@peraverse.lk"
    },
    {
      title: "Technical Support",
      contact: "+94 81 239 3001",
      email: "tech@peraverse.lk"
    },
    {
      title: "Registration Desk",
      contact: "+94 81 239 3002",
      email: "registration@peraverse.lk"
    }
  ];

  const handleViewOnMap = () => {
    alert("Navigating to Interactive Map Page with Phone Booth Locations");
  };

  const getTypeColorClass = (type: string): string => {
    switch(type.toLowerCase()) {
      case 'emergency': return 'bg-red-500/20 text-red-300 border-red-500/50';
      case 'security': return 'bg-blue-500/20 text-blue-300 border-blue-500/50';
      case 'police': return 'bg-indigo-500/20 text-indigo-300 border-indigo-500/50';
      case 'fire': return 'bg-orange-500/20 text-orange-300 border-orange-500/50';
      case 'medical': return 'bg-green-500/20 text-green-300 border-green-500/50';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/50';
    }
  };

  const renderEmergencyContacts = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {emergencyContacts.map((contact, index) => (
        <div key={index} className="bg-white/10 backdrop-blur-xl rounded-2xl border border-white/20 p-6 shadow-[0_8px_32px_rgba(0,0,0,0.3)] transition-all duration-300 hover:transform hover:-translate-y-1">
          <div className="flex justify-between items-start mb-4">
            <h3 className="text-white text-xl font-bold">{contact.title}</h3>
            <span className={`px-3 py-1 rounded-full text-xs font-bold border ${getTypeColorClass(contact.type)}`}>
              {contact.type}
            </span>
          </div>
          <div className="flex items-center gap-3 mb-4">
            <span className="text-2xl">📞</span>
            <span className="text-white text-2xl font-bold">{contact.number}</span>
          </div>
          <p className="text-white/80 mb-4">{contact.description}</p>
          <div className="flex items-center gap-2">
            <span className="text-lg">🕐</span>
            <span className="text-green-400 font-semibold">{contact.availability}</span>
          </div>
        </div>
      ))}
    </div>
  );

  const renderMedicalCenters = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {medicalCenters.map((center, index) => (
        <div key={index} className="bg-white/10 backdrop-blur-xl rounded-2xl border border-white/20 p-6 shadow-[0_8px_32px_rgba(0,0,0,0.3)] transition-all duration-300 hover:transform hover:-translate-y-1">
          <h3 className="text-white text-xl font-bold mb-4">{center.name}</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-lg mt-1">📍</span>
              <span className="text-white/90">{center.address}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-lg">📞</span>
              <span className="text-white/90">{center.phone}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-lg">🕐</span>
              <span className="text-green-400 font-semibold">{center.hours}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderPhoneBooths = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {phoneBooths.map((location, index) => (
          <div key={index} className="bg-white/10 backdrop-blur-xl rounded-xl border border-white/20 p-4 shadow-[0_8px_32px_rgba(0,0,0,0.3)] transition-all duration-300 hover:transform hover:-translate-y-1">
            <div className="flex items-center gap-4">
              <div className="bg-blue-500/20 p-3 rounded-full">
                <span className="text-2xl">📞</span>
              </div>
              <p className="text-white font-medium">{location}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="flex justify-center mt-8">
        <button 
          className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-semibold shadow-[0_6px_20px_rgba(59,130,246,0.4)] transition-all duration-300 hover:transform hover:-translate-y-1 hover:shadow-[0_8px_25px_rgba(59,130,246,0.5)]"
          onClick={handleViewOnMap}
        >
          <span className="text-xl">🗺️</span>
          View on Interactive Map
        </button>
      </div>
    </div>
  );

  const renderEventInfo = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {eventInfo.map((info, index) => (
        <div key={index} className="bg-white/10 backdrop-blur-xl rounded-2xl border border-white/20 p-6 shadow-[0_8px_32px_rgba(0,0,0,0.3)] transition-all duration-300 hover:transform hover:-translate-y-1">
          <h3 className="text-white text-xl font-bold mb-4">{info.title}</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-lg">📞</span>
              <span className="text-white/90">{info.contact}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-lg">📧</span>
              <span className="text-white/90 text-sm break-all">{info.email}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderImportantInfo = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-red-500/10 backdrop-blur-xl rounded-2xl border border-red-500/20 p-6 shadow-[0_8px_32px_rgba(0,0,0,0.3)]">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">🚨</span>
          <h3 className="text-white text-xl font-bold">Emergency Procedures</h3>
        </div>
        <ul className="space-y-2 text-white/90">
          <li className="flex items-start gap-2">
            <span className="text-red-400 mt-1">•</span>
            In case of fire, evacuate immediately and call 110
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-400 mt-1">•</span>
            For medical emergencies, call 119 or nearest medical center
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-400 mt-1">•</span>
            Report security issues to University Security immediately
          </li>
          <li className="flex items-start gap-2">
            <span className="text-red-400 mt-1">•</span>
            Assembly point: Main parking area near Engineering Faculty
          </li>
        </ul>
      </div>
      <div className="bg-blue-500/10 backdrop-blur-xl rounded-2xl border border-blue-500/20 p-6 shadow-[0_8px_32px_rgba(0,0,0,0.3)]">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">📋</span>
          <h3 className="text-white text-xl font-bold">Event Guidelines</h3>
        </div>
        <ul className="space-y-2 text-white/90">
          <li className="flex items-start gap-2">
            <span className="text-blue-400 mt-1">•</span>
            Please keep your registration badge visible at all times
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 mt-1">•</span>
            Follow all safety protocols and guidelines
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 mt-1">•</span>
            Report any suspicious activity to security
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 mt-1">•</span>
            Emergency exits are clearly marked throughout the venue
          </li>
        </ul>
      </div>
    </div>
  );

  const renderContent = () => {
    switch(selectedSection) {
      case 'emergency':
        return renderEmergencyContacts();
      case 'medical':
        return renderMedicalCenters();
      case 'phones':
        return renderPhoneBooths();
      case 'event':
        return renderEventInfo();
      case 'important':
        return renderImportantInfo();
      default:
        return renderEmergencyContacts();
    }
  };

  const getSectionInfo = () => {
    const sections = {
      emergency: { icon: '🚨', title: 'Emergency Contacts' },
      medical: { icon: '🏥', title: 'Medical Centers' },
      phones: { icon: '📞', title: 'Phone Booth Locations' },
      event: { icon: 'ℹ️', title: 'Event Information' },
      important: { icon: '📋', title: 'Important Information' }
    };
    return sections[selectedSection];
  };

  return (
    <div className="w-full h-full p-0 overflow-visible font-sans min-h-screen">
      <div className="max-w-6xl mx-auto flex flex-col gap-8 h-auto w-full px-4 py-8">
        {/* Header */}
        <div className="text-center mb-4 p-8 bg-white/5 rounded-2xl backdrop-blur-md border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.2)]">
          <div className="mb-4">
            <span className="text-5xl inline-block p-4 bg-gradient-to-br from-blue-400 to-blue-600 rounded-full shadow-[0_4px_20px_rgba(96,165,250,0.4)]">
              📞
            </span>
          </div>
          <h1 className="text-4xl text-white mb-2 font-extrabold drop-shadow-[0_2px_4px_rgba(0,0,0,0.3)] bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
            Emergency & Contact Information
          </h1>
          <p className="text-slate-300 text-xl font-medium m-0">
            Faculty of Engineering • University of Peradeniya
          </p>
        </div>

        {/* Section Buttons */}
        <div className="flex justify-center gap-4 mb-8 flex-wrap">
          {[
            { key: 'emergency' as const, icon: '🚨', label: 'Emergency Contacts' },
            { key: 'medical' as const, icon: '🏥', label: 'Medical Centers' },
            { key: 'phones' as const, icon: '📞', label: 'Phone Booths' },
            { key: 'event' as const, icon: 'ℹ️', label: 'Event Info' },
            { key: 'important' as const, icon: '📋', label: 'Important Info' }
          ].map((section) => (
            <button
              key={section.key}
              className={`flex items-center gap-2 px-6 py-4 rounded-xl text-base font-semibold cursor-pointer transition-all duration-300 backdrop-blur-md ${
                selectedSection === section.key
                  ? 'bg-gradient-to-br from-blue-400 to-blue-600 text-white border border-blue-400 shadow-[0_6px_20px_rgba(96,165,250,0.4)]'
                  : 'bg-blue-400/10 text-slate-200 border border-blue-400/20 hover:bg-blue-400/20 hover:-translate-y-1 hover:shadow-[0_6px_20px_rgba(96,165,250,0.2)]'
              }`}
              onClick={() => setSelectedSection(section.key)}
            >
              <span>{section.icon}</span>
              {section.label}
            </button>
          ))}
        </div>

        {/* Section Content */}
        <div className="space-y-6">
          <div className="flex items-center gap-4 mb-6">
            <span className="text-4xl">{getSectionInfo().icon}</span>
            <h2 className="text-3xl font-bold text-white">{getSectionInfo().title}</h2>
          </div>
          {renderContent()}
        </div>
      </div>
    </div>
  );
};

export default ContactPageTailwind;
