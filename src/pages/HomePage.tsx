import React, { useEffect, useState } from 'react';
import VideoHero from '../components/VideoHero';
import UniversityInfo from '../components/UniversityInfo';
import { ChevronDown, Award, Users, BookOpen, Globe, type LucideIcon } from 'lucide-react';

interface UniversityFeature {
  id: number;
  icon: LucideIcon;
  title: string;
  description: string;
  color: string;
}

const HomePage: React.FC = () => {
  const isLoaded  = useState<boolean>(false);
  const showScrollHint = useState<boolean>(true);
  
  // University data configuration
  const universityData = {
    foundedYear: 1950,
    currentYear: new Date().getFullYear(),
    name: 'University of Peradeniya',
    location: 'Peradeniya, Sri Lanka',
    studentCount: '12,000+',
    facultyCount: '800+',
    departments: 9,
    achievements: [
      'Top University in Sri Lanka',
      'Excellence in Engineering & Science',
      'Outstanding Research Publications',
      'Strong Alumni Network Globally'
    ],
    description: 'The University of Peradeniya stands as the pinnacle of higher education in Sri Lanka. We continue to nurture brilliant minds, conduct groundbreaking research, and contribute to the development of our nation and the world.',
    milestones: [
      { year: 1950, event: 'University of Peradeniya Established' },
      { year: 1965, event: 'First Engineering Faculty Founded' },
      { year: 1980, event: 'Research Excellence Recognition' },
      { year: 2000, event: 'Digital Campus Initiative' },
      { year: 2020, event: 'International Accreditation Achieved' }
    ]
  };

  // University features and highlights
  const universityFeatures: UniversityFeature[] = [
    {
      id: 1,
      icon: Award,
      title: 'Academic Excellence',
      description: 'Recognized as the premier university in Sri Lanka with world-class academic programs and faculty.',
      color: '#3498db'
    },
    {
      id: 2,
      icon: BookOpen,
      title: 'Research Innovation',
      description: 'Leading groundbreaking research across multiple disciplines with state-of-the-art facilities.',
      color: '#e74c3c'
    },
    {
      id: 3,
      icon: Users,
      title: 'Global Community',
      description: 'A diverse community of students, faculty, and alumni making impact worldwide.',
      color: '#f39c12'
    },
    {
      id: 4,
      icon: Globe,
      title: 'International Recognition',
      description: 'Partnerships with leading universities globally and internationally accredited programs.',
      color: '#27ae60'
    }
  ];

  const scrollToNext = (): void => {
    const nextSection = document.getElementById('university-info');
    if (nextSection) {
      nextSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <>
      <div className={`min-h-screen overflow-hidden transition-opacity duration-500 ease-in-out ${isLoaded ? 'opacity-100' : 'opacity-0'}`}>
        {/* Hero Section with Video */}
        <section className="relative h-screen">
          <VideoHero
            videoSrc="/video/video.mp4"
            posterImage="/images/university-campus.jpg"
            title="University of Peradeniya"
            subtitle="Excellence in Education, Research & Innovation"
            autoPlay={true}
            showControls={false}
          />
          {/* Scroll Hint */}
          {showScrollHint && (
            <div 
              className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex flex-col items-center gap-2 cursor-pointer text-white z-10 animate-bounce"
              onClick={scrollToNext}
            >
              <span className="text-base font-medium text-shadow-sm">Explore Our University</span>
              <ChevronDown className="animate-bounce" />
            </div>
          )}
        </section>

        {/* University Information Section */}
        <section id="university-info" className="relative z-[2]">
          <UniversityInfo 
            data={universityData}
            layout="default"
          />
        </section>

        {/* University Features Section */}
        <section id="features" className="bg-gray-50 py-16 px-8 relative z-[2]">
          <div className="max-w-6xl mx-auto text-center">
            <h2 className="text-4xl font-bold mb-12 text-slate-800 md:text-3xl sm:text-2xl">
              Why Choose University of Peradeniya
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 auto-fit-minmax-280">
              {universityFeatures.map((feature) => {
                const IconComponent = feature.icon;
                return (
                  <div 
                    key={feature.id} 
                    className="bg-white p-8 rounded-xl shadow-md text-center transition-all duration-300 ease-in-out hover:-translate-y-1 hover:shadow-lg"
                  >
                    <div 
                      className="w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6"
                      style={{backgroundColor: feature.color}}
                    >
                      <IconComponent size={32} color="white" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-800 mb-4">
                      {feature.title}
                    </h3>
                    <p className="text-gray-500 leading-relaxed text-base">
                      {feature.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      </div>

      {/* Custom CSS for animations and specific styles */}
      <style>{`
        @keyframes bounce {
          0%, 20%, 50%, 80%, 100% { 
            transform: translateY(0); 
          }
          40% { 
            transform: translateY(-10px); 
          }
          60% { 
            transform: translateY(-5px); 
          }
        }

        @keyframes sparkle {
          0%, 100% { 
            transform: scale(1) rotate(0deg); 
            opacity: 1; 
          }
          50% { 
            transform: scale(1.2) rotate(180deg); 
            opacity: 0.8; 
          }
        }

        .text-shadow-sm {
          text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }

        .auto-fit-minmax-280 {
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        }

        @media (max-width: 768px) {
          .auto-fit-minmax-280 {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 480px) {
          .auto-fit-minmax-280 {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </>
  );
};

export default HomePage;
