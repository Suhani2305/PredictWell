import React, { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../../context/ThemeContext';
import { FiActivity, FiCpu } from 'react-icons/fi';
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
  subtitle = 'Healthcare Intelligence',
  icon,
  iconColor,
  children,
  predictionResult
}) => {
  const { theme, accentColor } = useTheme();

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
    <div className="w-full max-w-[1400px] mx-auto px-4 py-8">
      {/* Page Title */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-16"
      >
        <TitleSection
          accentColor={accentColor}
          theme={theme}
          title={title}
          subtitlePrefix="Quick Check"
          subtitles={[
            'See your health risks easily',
            'Get reports instantly',
            'Simple health screening'
          ]}
        />
      </motion.div>

      {/* Main Container Card */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: 'spring', damping: 25, stiffness: 100 }}
        className={`relative rounded-[3rem] overflow-hidden border ${theme === 'dark'
          ? 'bg-[#050507] border-white/5 shadow-none'
          : 'bg-white border-slate-200/60 shadow-[0_40px_100px_-20px_rgba(0,0,0,0.08)]'
          }`}
      >
        {/* Dynamic Inner Glow */}
        <div className="absolute top-0 right-0 w-[500px] h-[500px] blur-[150px] opacity-[0.03] pointer-events-none"
          style={{ background: accentColor }} />

        <div className="grid grid-cols-1 lg:grid-cols-2 relative z-10">

          {/* Left Panel: Form Input */}
          <div className={`p-8 md:p-14 border-b lg:border-b-0 lg:border-r ${theme === 'dark' ? 'bg-[#0f0f12] border-white/5' : 'bg-slate-50/50 border-slate-100'
            }`}>
            <div className="flex flex-col gap-8 mb-12">
              <div className="flex items-center justify-between">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#14b8a6]/10 text-[#14b8a6] text-[10px] font-black uppercase tracking-widest border border-[#14b8a6]/20">
                  <FiActivity size={12} /> Diagnostic Input
                </div>
                <div className="text-[10px] font-black uppercase tracking-widest opacity-30">v1.2.0-secure</div>
              </div>

              <div className="flex items-center gap-5">
                <div className="w-16 h-16 rounded-2xl flex items-center justify-center shadow-2xl transition-transform hover:scale-105 active:scale-95"
                  style={{
                    background: `linear-gradient(135deg, ${accentColor}, #0ea5e9)`,
                    color: 'white',
                    boxShadow: `0 20px 40px -10px ${accentColor}40`
                  }}>
                  {icon}
                </div>
                <div>
                  <h3 className="text-3xl font-black tracking-tighter">Medical Parameters</h3>
                  <p className="text-xs opacity-50 font-bold tracking-[0.2em] uppercase mt-1">Symmetrical Clinical Analysis</p>
                </div>
              </div>
            </div>

            <div className="space-y-8">
              {formElements.length > 0 ? formElements : children}
            </div>
          </div>

          {/* Right Panel: Results View */}
          <div className="p-8 md:p-14 flex flex-col bg-transparent">
            <div className="flex flex-col gap-8 mb-12">
              <div className="flex items-center justify-between">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 text-indigo-500 text-[10px] font-black uppercase tracking-widest border border-indigo-500/20">
                  <FiCpu size={12} /> Neural Engine Output
                </div>
              </div>

              <div className="flex items-center gap-5">
                <div className="w-2 h-12 rounded-full" style={{ background: accentColor }} />
                <div>
                  <h3 className="text-3xl font-black tracking-tighter">Analysis Terminal</h3>
                  <p className="text-xs opacity-50 font-bold tracking-[0.2em] uppercase mt-1">Real-time Prediction Result</p>
                </div>
              </div>
            </div>

            <div className="flex-1 flex flex-col">
              {predictionResult ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="h-full"
                >
                  {predictionResult}
                </motion.div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center space-y-8 py-20 border-2 border-dashed rounded-[3rem] border-slate-200 dark:border-white/5 bg-slate-50/20 dark:bg-white/[0.01]">
                  <div className="relative">
                    <div className="absolute inset-0 bg-[#14b8a6] blur-3xl opacity-20 animate-pulse" />
                    <div className="relative text-7xl filter grayscale opacity-40">🧬</div>
                  </div>
                  <div className="space-y-3">
                    <h4 className="text-2xl font-black tracking-tight">System Idle</h4>
                    <p className="max-w-xs text-sm font-medium opacity-50 leading-relaxed">
                      AI processing units are on standby. Please finalize the medical parameters on the left to initialize analysis.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      </motion.div>

      {/* Warning/Disclaimer Strip */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="mt-12 p-6 rounded-3xl bg-slate-100/50 dark:bg-white/5 border border-slate-200/50 dark:border-white/5 flex items-center gap-4 max-w-4xl mx-auto"
      >
        <div className="w-10 h-10 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center text-amber-600 shrink-0 font-bold">!</div>
        <p className="text-xs md:text-sm font-medium text-slate-500 dark:text-slate-400">
          <strong>Important Medical Disclaimer:</strong> This system provides AI-generated health assessments based on patterns in clinical data. It is intended as a screening tool only and does not constitute professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional.
        </p>
      </motion.div>
    </div>
  );
};

export default FormContainer;
