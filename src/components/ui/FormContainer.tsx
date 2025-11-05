import React, { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../../context/ThemeContext';
import TitleSection from './TitleSection';

interface FormContainerProps {
  title: string;
  subtitle?: string;
  icon: ReactNode;
  iconColor?: string;
  children: ReactNode;
  predictionResult?: ReactNode;
}

const FormContainer: React.FC<FormContainerProps> = ({
  title,
  subtitle = 'Prediction Model',
  icon,
  iconColor,
  children,
  predictionResult
}) => {
  const { theme, accentColor } = useTheme();
  
  // Enhanced animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.2
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 30, scale: 0.95 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        type: 'spring',
        stiffness: 120,
        damping: 18,
        mass: 0.8
      }
    }
  };

  // Find form and non-form children
  const formElements: React.ReactNode[] = [];
  const resultElements: React.ReactNode[] = [];
  
  React.Children.forEach(children, child => {
    if (React.isValidElement(child) && child.type === 'form') {
      formElements.push(child);
    } else {
      resultElements.push(child);
    }
  });

  return (
    <div className="container mx-auto px-4 py-12 flex flex-col items-center min-h-screen">
      {/* Enhanced Title Section */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="mb-12 mt-8 text-center w-full max-w-4xl"
      >
        <TitleSection 
          accentColor={accentColor}
          theme={theme}
          title={title}
          subtitlePrefix="AI-Powered"
          subtitles={[subtitle, 'Health Analysis', 'Prediction System']}
        />
      </motion.div>
      
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="max-w-7xl mx-auto w-full"
      >
        {/* Modern Card Container */}
        <motion.div 
          variants={itemVariants}
          className={`relative rounded-3xl overflow-hidden ${
            theme === 'dark' 
              ? 'bg-gradient-to-br from-gray-900/90 via-gray-800/80 to-gray-900/90' 
              : 'bg-gradient-to-br from-white/95 via-white/90 to-white/95'
          } backdrop-blur-2xl border`}
          style={{ 
            borderColor: `${accentColor}40`,
            boxShadow: `0 20px 60px -12px ${accentColor}30, 0 0 0 1px ${accentColor}20`
          }}
        >
          {/* Animated Background Gradient */}
          <div 
            className="absolute inset-0 opacity-30"
            style={{
              background: `radial-gradient(circle at 30% 50%, ${accentColor}20, transparent 50%),
                          radial-gradient(circle at 70% 80%, ${accentColor}15, transparent 50%)`,
              animation: 'gradient-shift 15s ease infinite'
            }}
          />

          {/* Icon Header */}
          <div className="relative flex flex-col items-center pt-8 pb-6 px-6 z-10">
            <motion.div 
              className="relative mb-4"
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ 
                type: 'spring',
                stiffness: 200,
                damping: 15,
                delay: 0.3
              }}
              whileHover={{ 
                scale: 1.1,
                rotate: 5,
                transition: { duration: 0.3 }
              }}
            >
              <div 
                className="w-20 h-20 rounded-2xl flex items-center justify-center relative overflow-hidden"
                style={{ 
                  background: `linear-gradient(135deg, ${accentColor}40, ${accentColor}20)`,
                  border: `2px solid ${accentColor}60`,
                  boxShadow: `0 8px 32px ${accentColor}40, inset 0 1px 0 ${accentColor}50`
                }}
              >
                {/* Shine Effect */}
                <div 
                  className="absolute inset-0"
                  style={{
                    background: `linear-gradient(135deg, transparent 40%, ${accentColor}30 50%, transparent 60%)`,
                    animation: 'shimmer 3s ease-in-out infinite'
                  }}
                />
                <div style={{ color: accentColor, zIndex: 1, position: 'relative' }}>
                  {icon}
                </div>
              </div>
            </motion.div>
          </div>
          
          {/* Two Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 relative z-10">
            {/* Form Column */}
            <motion.div 
              variants={itemVariants}
              className="flex flex-col"
            >
              <div 
                className={`rounded-2xl p-6 h-full flex flex-col border transition-all duration-300 ${
                  theme === 'dark' 
                    ? 'bg-gradient-to-br from-gray-900/60 to-gray-800/40' 
                    : 'bg-gradient-to-br from-white/60 to-gray-50/40'
                } backdrop-blur-xl`}
                style={{ 
                  borderColor: `${accentColor}40`,
                  boxShadow: `0 8px 24px -4px ${accentColor}20, inset 0 1px 0 ${accentColor}20`
                }}
              >
                <motion.h2 
                  className="text-xl font-bold mb-6 flex items-center gap-3"
                  style={{ color: accentColor }}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <div 
                    className="w-1 h-6 rounded-full"
                    style={{ backgroundColor: accentColor }}
                  />
                  Input Data
                </motion.h2>
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                  {formElements.length > 0 ? formElements : children}
                </div>
              </div>
            </motion.div>
            
            {/* Prediction Result Column */}
            <motion.div 
              variants={itemVariants}
              className="flex flex-col"
            >
              <div 
                className={`rounded-2xl p-6 h-full flex flex-col border transition-all duration-300 ${
                  theme === 'dark' 
                    ? 'bg-gradient-to-br from-gray-900/60 to-gray-800/40' 
                    : 'bg-gradient-to-br from-white/60 to-gray-50/40'
                } backdrop-blur-xl`}
                style={{ 
                  borderColor: `${accentColor}40`,
                  boxShadow: `0 8px 24px -4px ${accentColor}20, inset 0 1px 0 ${accentColor}20`
                }}
              >
                <motion.h2 
                  className="text-xl font-bold mb-6 flex items-center gap-3"
                  style={{ color: accentColor }}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <div 
                    className="w-1 h-6 rounded-full"
                    style={{ backgroundColor: accentColor }}
                  />
                  Prediction Results
                </motion.h2>
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                  {resultElements.length > 0 ? resultElements : predictionResult || (
                    <div className="flex items-center justify-center h-full text-center">
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className={`p-8 rounded-xl ${
                          theme === 'dark' ? 'bg-gray-800/30' : 'bg-gray-100/30'
                        }`}
                      >
                        <div className={`text-lg mb-2 ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}`}>
                          Enter data and click predict
                        </div>
                        <div className={`text-sm ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
                          Results will appear here
                        </div>
                      </motion.div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </motion.div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: ${theme === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'};
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: ${accentColor}40;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: ${accentColor}60;
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
          100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        @keyframes gradient-shift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
      `}</style>
    </div>
  );
};

export default FormContainer;
