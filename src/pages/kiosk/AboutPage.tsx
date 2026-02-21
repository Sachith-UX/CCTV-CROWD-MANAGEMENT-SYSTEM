import React from "react";
import { motion } from "framer-motion";
import logo from "./kioskAssets/engex.png";

interface EventStat {
  label: string;
  value: string;
  icon: string;
}

interface Department {
  name: string;
  description: string;
  projects: number;
  color: string;
}

interface Highlight {
  icon: string;
  text: string;
}

interface AboutPageTailwindProps {}

const AboutPageTailwind: React.FC<AboutPageTailwindProps> = () => {
  const eventStats: EventStat[] = [
    { label: "Engineering Departments", value: "8", icon: "🏛️" },
    { label: "Student Projects", value: "120+", icon: "🚀" },
    { label: "Industry Partners", value: "45", icon: "🤝" },
    { label: "Innovation Exhibits", value: "75", icon: "💡" },
    { label: "Research Papers", value: "200+", icon: "📄" },
    { label: "Guest Speakers", value: "25", icon: "🎤" }
  ];

  const departments: Department[] = [
    { name: "Computer Engineering", description: "AI, Software, Cybersecurity", projects: 18, color: "#3b82f6" },
    { name: "Electrical Engineering", description: "Power Systems, Renewable Energy, Smart Grids", projects: 16, color: "#f59e0b" },
    { name: "Mechanical Engineering", description: "Robotics, Manufacturing, Automotive Technology", projects: 15, color: "#10b981" },
    { name: "Civil Engineering", description: "Infrastructure, Construction, Environmental Engineering", projects: 14, color: "#8b5cf6" },
    { name: "Chemical & Process Engineering", description: "Process Engineering, Materials Science, Sustainability", projects: 12, color: "#ef4444" },
    { name: "Manufacturing & Industrial Engineering", description: "Production, Automation, Quality Control", projects: 11, color: "#06b6d4" },
    { name: "Engineering Management", description: "Project Leadership, Operations Optimization, Tech Management", projects: 11, color: "#d406a7" },
    { name: "Engineering Mathematics", description: "Computational Methods, Modeling, Data Analysis", projects: 11, color: "#d3d15e" }
  ];

  const highlights: Highlight[] = [
    { icon: "🏆", text: "Award Competitions" },
    { icon: "🎯", text: "Interactive Demos" },
    { icon: "🔬", text: "Research Presentations" },
    { icon: "🤖", text: "Technology Showcases" },
    { icon: "💼", text: "Industry Partnerships" },
    { icon: "🌱", text: "Sustainability Focus" }
  ];

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto flex flex-col gap-8">

        {/* Header */}
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <div className="flex items-center gap-5 mb-5">
            <motion.img 
              src={logo} 
              alt="EngEx 2025 Logo" 
              className="w-[350px] h-[350px] object-contain rounded-lg transition-transform duration-300 hover:scale-105 hover:rotate-3"
              whileHover={{ scale: 1.05, rotate: 3 }}
              transition={{ type: "spring", stiffness: 200 }}
            />
            <div className="flex-1">
              <h1 className="text-8xl font-bold text-white mb-4 flex items-center justify-center gap-4">
                EngEx 2025
              </h1>
              <p className="text-blue-300 text-3xl font-semibold mb-6 drop-shadow-[0_0_10px_rgba(147,197,253,0.7)]">
                The Biggest Engineering Exhibition Of The Year
              </p>
            </div>
          </div>
        </motion.div>

        {/* Event Overview */}
        <motion.div 
          className="backdrop-blur-2xl bg-white/10 rounded-2xl border border-white/20 p-8 shadow-[0_25px_50px_-12px_rgba(0,0,0,0.5)] transition-all duration-300 hover:-translate-y-1"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <h2 className="text-3xl font-bold text-white mb-8 text-center flex items-center justify-center gap-3">
            Event Overview
          </h2>
          <div className="flex flex-col gap-8">
            <div className="space-y-4">
              <p className="text-white/90 text-lg leading-7 m-0">
                EngEx 2025 is the premier engineering exhibition showcasing innovative 
                solutions and cutting-edge research from the Faculty of Engineering, 
                University of Peradeniya.
              </p>
              <p className="text-white/90 text-lg leading-7 m-0">
                This year's theme: <b>"Engineering for a Sustainable Future"</b> highlights
                projects addressing global challenges in sustainability, technology, and human welfare.
              </p>
            </div>

            {/* Highlights Grid */}
            <motion.div 
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4"
              initial="hidden"
              whileInView="visible"
              variants={{
                hidden: {},
                visible: {
                  transition: {
                    staggerChildren: 0.1
                  }
                }
              }}
            >
              {highlights.map((highlight, i) => (
                <motion.div 
                  key={i} 
                  className="flex items-center gap-3 p-4 bg-white/10 rounded-xl border border-white/20 transition-all duration-300 hover:bg-gradient-to-br hover:from-blue-400/30 hover:to-purple-400/30 hover:-translate-y-1 hover:scale-105 hover:shadow-[0_0_15px_rgba(96,165,250,0.4)]"
                  variants={{
                    hidden: { opacity: 0, y: 20 },
                    visible: { opacity: 1, y: 0 }
                  }}
                >
                  <div className="text-2xl flex-shrink-0">
                    {highlight.icon}
                  </div>
                  <div className="text-white font-medium">
                    {highlight.text}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </motion.div>

        {/* Departments */}
        <div className="backdrop-blur-2xl bg-white/10 rounded-2xl border border-white/20 p-8 shadow-[0_25px_50px_-12px_rgba(0,0,0,0.5)]">
          <h2 className="text-3xl font-bold text-white mb-8 text-center flex items-center justify-center gap-3">
            🏛️ Engineering Departments
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {departments.map((dept, i) => (
              <motion.div 
                key={i} 
                className="p-6 bg-white/10 rounded-xl border border-white/20 transition-all duration-300 hover:scale-105 hover:-translate-y-1"
                whileHover={{ scale: 1.08, y: -5 }}
                transition={{ type: "spring", stiffness: 150 }}
                style={{ borderLeft: `5px solid ${dept.color}` }}
              >
                <h3 className="text-white font-bold text-lg mb-2">{dept.name}</h3>
                <span className="text-blue-300 text-sm font-medium mb-3 block">{dept.projects} projects</span>
                <p className="text-white/80 text-sm leading-relaxed">{dept.description}</p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {eventStats.map((stat, i) => (
            <motion.div 
              key={i} 
              className="backdrop-blur-2xl bg-white/10 rounded-2xl border border-white/20 p-6 text-center shadow-[0_25px_50px_-12px_rgba(0,0,0,0.5)] transition-all duration-300 hover:scale-110"
              whileHover={{ scale: 1.1 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-3xl mb-2">{stat.icon}</div>
              <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
              <div className="text-white/80 text-sm font-medium">{stat.label}</div>
            </motion.div>
          ))}
        </div>

      </div>
    </div>
  );
};

export default AboutPageTailwind;
