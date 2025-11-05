import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import TitleSection from '../components/ui/TitleSection';
import Navbar from '../components/layout/Navbar';
import MainLayout from '../components/layout/MainLayout';
import { useRouter } from 'next/router';
import { FiActivity, FiHeart, FiDroplet, FiThermometer, FiSun, FiInfo, FiCrosshair, FiArrowRight } from 'react-icons/fi';
import { RiVirusFill } from 'react-icons/ri';
import Image from 'next/image';

// Define model card data with enhanced metadata
const modelCards = [
  {
    id: 'symptom',
    title: 'Symptom to Disease',
    description: 'Predict possible diseases based on your symptoms',
    image: '/sample.webp',
    route: '/symptom',
    icon: <RiVirusFill size={32} />,
    gradient: 'from-purple-500/20 to-pink-500/20'
  },
  {
    id: 'heart',
    title: 'Heart Disease',
    description: 'Check your heart health with our prediction model',
    image: '/heart.png',
    route: '/heart',
    icon: <FiHeart size={32} />,
    gradient: 'from-red-500/20 to-orange-500/20'
  },
  {
    id: 'liver',
    title: 'Liver Disease',
    description: 'Analyze liver function parameters to predict disease',
    image: '/liver.png',
    route: '/liver',
    icon: <FiDroplet size={32} />,
    gradient: 'from-blue-500/20 to-cyan-500/20'
  },
  {
    id: 'diabetes',
    title: 'Diabetes',
    description: 'Predict diabetes risk based on health parameters',
    image: '/diabetes.png',
    route: '/diabetes',
    icon: <FiThermometer size={32} />,
    gradient: 'from-yellow-500/20 to-orange-500/20'
  },
  {
    id: 'skin',
    title: 'Skin Cancer',
    description: 'Analyze skin lesions to detect potential skin cancer',
    image: '/skin.png',
    route: '/skin',
    icon: <FiSun size={32} />,
    gradient: 'from-green-500/20 to-emerald-500/20'
  },
  {
    id: 'breast',
    title: 'Breast Cancer',
    description: 'Analyze breast cancer risk based on health parameters',
    image: '/breast.png',
    route: '/breast',
    icon: <FiCrosshair size={32} />,
    gradient: 'from-pink-500/20 to-rose-500/20'
  },
  {
    id: 'about',
    title: 'About',
    description: 'Learn more about our models and how they work',
    image: '/about.webp',
    route: '/about',
    icon: <FiInfo size={32} />,
    gradient: 'from-indigo-500/20 to-purple-500/20'
  }
];

const HomePage: React.FC = () => {
  const { theme, accentColor } = useTheme();
  const router = useRouter();

  // Enhanced animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08,
        delayChildren: 0.2
      }
    }
  };

  const cardVariants = {
    hidden: { 
      y: 40, 
      opacity: 0,
      scale: 0.95
    },
    visible: { 
      y: 0, 
      opacity: 1,
      scale: 1,
      transition: {
        type: 'spring',
        stiffness: 120,
        damping: 15,
        mass: 0.8
      }
    },
    hover: {
      y: -12,
      scale: 1.03,
      transition: {
        type: 'spring',
        stiffness: 400,
        damping: 15,
        mass: 0.5
      }
    },
    tap: {
      scale: 0.98,
      transition: {
        type: 'spring',
        stiffness: 400,
        damping: 15
      }
    }
  };

  return (
    <>
      <Navbar />
      <MainLayout>
        <div className="container mx-auto px-4 py-12 flex flex-col items-center min-h-screen">
          {/* Hero Section */}
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="mb-16 mt-8 text-center w-full max-w-4xl"
          >
            <TitleSection 
              accentColor={accentColor}
              theme={theme}
              title="PredictWell"
              subtitlePrefix="Your"
              subtitles={[
                'Disease Prediction System',
                'Health Analysis Platform',
                'Medical Assistant',
                'Diagnostic Helper'
              ]}
            />
            <motion.p 
              className={`mt-6 text-lg ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'} max-w-2xl mx-auto`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              Advanced AI-powered health prediction models for accurate disease diagnosis and risk assessment
            </motion.p>
          </motion.div>

          {/* Cards Grid - All cards in modern grid */}
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 max-w-7xl mx-auto w-full px-4"
          >
            {modelCards.map((card, index) => (
              <motion.div
                key={card.id}
                className="group relative cursor-pointer h-full"
                variants={cardVariants}
                whileHover="hover"
                whileTap="tap"
                onClick={() => router.push(card.route)}
                style={{
                  filter: `drop-shadow(0 4px 6px ${accentColor}15)`
                }}
              >
                {/* Modern Card with Gradient Border */}
                <div 
                  className={`relative h-full rounded-2xl overflow-hidden transition-all duration-500 ${
                    theme === 'dark' 
                      ? 'bg-gradient-to-br from-gray-900/80 via-gray-800/70 to-gray-900/80' 
                      : 'bg-gradient-to-br from-white/90 via-white/80 to-white/90'
                  } backdrop-blur-xl border`}
                  style={{ 
                    borderColor: `${accentColor}30`,
                    boxShadow: `0 8px 32px -8px ${accentColor}30, 0 0 0 1px ${accentColor}10`
                  }}
                >
                  {/* Animated Gradient Overlay */}
                  <div 
                    className={`absolute inset-0 bg-gradient-to-br ${card.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}
                  />

                  {/* Accent Border Glow on Hover */}
                  <div 
                    className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                    style={{
                      background: `linear-gradient(135deg, ${accentColor}20, transparent, ${accentColor}10)`,
                      boxShadow: `inset 0 0 40px ${accentColor}20`
                    }}
                  />

                  {/* Content */}
                  <div className="relative p-8 flex flex-col h-full items-center text-center z-10">
                    {/* Icon Container with Modern Design */}
                    <motion.div 
                      className="relative mb-6"
                      initial={{ rotate: 0, scale: 1 }}
                      animate={{ 
                        rotate: [0, -5, 5, -5, 0],
                        scale: [1, 1.02, 1, 1.02, 1]
                      }}
                      transition={{ 
                        repeat: Infinity, 
                        duration: 4,
                        ease: "easeInOut"
                      }}
                      whileHover={{ 
                        scale: 1.15,
                        rotate: [0, -10, 10, -10, 0],
                        transition: { duration: 0.6 } 
                      }}
                    >
                      {/* Icon Background Circle */}
                      <div 
                        className="w-24 h-24 rounded-2xl flex items-center justify-center relative overflow-hidden"
                        style={{ 
                          background: `linear-gradient(135deg, ${accentColor}30, ${accentColor}15)`,
                          boxShadow: `0 8px 24px ${accentColor}30, inset 0 1px 0 ${accentColor}40`
                        }}
                      >
                        {/* Shine effect */}
                        <div 
                          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                          style={{
                            background: `linear-gradient(135deg, transparent 30%, ${accentColor}40 50%, transparent 70%)`,
                            transform: 'translateX(-100%)',
                            animation: 'shimmer 1.5s ease-in-out'
                          }}
                        />
                        <div 
                          style={{ color: accentColor }}
                          className="relative z-10"
                        >
                          {card.icon}
                        </div>
                      </div>
                      
                      {/* Floating particles */}
                      <div className="absolute inset-0 pointer-events-none">
                        {[...Array(3)].map((_, i) => (
                          <motion.div
                            key={i}
                            className="absolute w-1 h-1 rounded-full"
                            style={{ 
                              backgroundColor: accentColor,
                              left: `${30 + i * 20}%`,
                              top: `${20 + i * 30}%`
                            }}
                            animate={{
                              y: [0, -20, 0],
                              opacity: [0, 0.8, 0],
                              scale: [0, 1, 0]
                            }}
                            transition={{
                              duration: 2,
                              repeat: Infinity,
                              delay: i * 0.3,
                              ease: "easeInOut"
                            }}
                          />
                        ))}
                      </div>
                    </motion.div>
                    
                    {/* Title */}
                    <motion.h3 
                      className="text-2xl font-bold mb-3 tracking-tight"
                      style={{ color: accentColor }}
                      whileHover={{ scale: 1.05 }}
                      transition={{ duration: 0.2 }}
                    >
                      {card.title}
                    </motion.h3>
                    
                    {/* Description */}
                    <p className={`text-sm leading-relaxed mb-6 flex-grow ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'}`}>
                      {card.description}
                    </p>
                    
                    {/* CTA Button */}
                    <motion.div 
                      className="w-full mt-auto"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div 
                        className="flex items-center justify-center gap-2 py-3 px-6 rounded-xl font-medium text-sm transition-all duration-300"
                        style={{ 
                          color: accentColor,
                          backgroundColor: `${accentColor}15`,
                          border: `1px solid ${accentColor}30`,
                          boxShadow: `0 4px 12px ${accentColor}20`
                        }}
                      >
                        <span>Explore</span>
                        <motion.div
                          animate={{ x: [0, 4, 0] }}
                          transition={{ 
                            duration: 1.5,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        >
                          <FiArrowRight size={16} />
                        </motion.div>
                      </div>
                    </motion.div>
                  </div>

                  {/* Bottom Accent Line */}
                  <div 
                    className="absolute bottom-0 left-0 right-0 h-1 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                    style={{
                      background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)`,
                      boxShadow: `0 0 10px ${accentColor}`
                    }}
                  />
                </div>
              </motion.div>
            ))}
          </motion.div>

          {/* Footer Info */}
          <motion.div 
            className="mt-16 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            <p className={`text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
              Powered by advanced machine learning models
            </p>
          </motion.div>
        </div>
      </MainLayout>

      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
          100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
      `}</style>
    </>
  );
};

export default HomePage;
