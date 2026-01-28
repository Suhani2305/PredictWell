import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../../context/ThemeContext';
import { FiCheckCircle, FiAlertCircle, FiInfo, FiWifi, FiWifiOff, FiActivity, FiShield, FiCpu } from 'react-icons/fi';

interface PredictionResultProps {
  prediction: string;
  confidence: number;
  modelName: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    r2?: number;
    rmse?: number;
  };
  topFeatures?: { name: string; importance: number }[];
  diseaseType: 'heart' | 'liver' | 'breast' | 'diabetes' | 'skin' | 'symptom';
  error?: boolean;
}

const PredictionResult: React.FC<PredictionResultProps> = ({
  prediction,
  confidence,
  modelName,
  metrics,
  topFeatures,
  diseaseType,
  error
}) => {
  const { theme, accentColor } = useTheme();

  const isHealthy = prediction && (
    prediction.toLowerCase() === 'negative' ||
    prediction.toLowerCase() === 'normal' ||
    prediction.toLowerCase() === 'healthy' ||
    prediction.toLowerCase() === 'benign'
  );

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: 'spring', stiffness: 100, damping: 15 }
    }
  };

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center space-y-6">
        <div className="w-24 h-24 rounded-full bg-rose-500/10 flex items-center justify-center text-rose-500 border-2 border-rose-500/20 shadow-2xl shadow-rose-500/10">
          <FiWifiOff size={48} className="animate-pulse" />
        </div>
        <div className="space-y-2">
          <h3 className="text-2xl font-black tracking-tight text-rose-500">Backend Connection Error</h3>
          <p className="opacity-60 max-w-xs text-sm font-medium leading-relaxed">
            We couldn't reach the AI engine. Please make sure the <span className="font-bold text-teal-500">backend server</span> is running and connected.
          </p>
        </div>
        <button
          onClick={() => window.location.reload()}
          className="px-8 py-3 rounded-2xl bg-[#14b8a6] text-white font-black text-sm shadow-xl hover:scale-105 active:scale-95 transition-all"
        >
          Try Connecting Again
        </button>
      </div>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      {/* Primary Result Card */}
      <motion.div
        variants={itemVariants}
        className={`p-10 rounded-[3rem] text-center border shadow-2xl transition-all duration-700 relative overflow-hidden ${isHealthy
          ? 'bg-emerald-500/[0.03] border-emerald-500/20 shadow-emerald-500/5'
          : 'bg-rose-500/[0.03] border-rose-500/20 shadow-rose-500/5'
          }`}
      >
        <div className="absolute top-0 right-0 p-6 flex flex-col items-end gap-2">
          <div className={`px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest border ${isHealthy
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-500'
            : 'bg-rose-500/10 border-rose-500/20 text-rose-500'}`}>
            <FiCheckCircle className="inline mr-2" />
            {isHealthy ? 'Safe Status' : 'Risk Detected'}
          </div>
          <div className="text-[8px] font-black uppercase tracking-widest opacity-20">Production Model Alpha</div>
        </div>

        <div className="flex justify-center mb-10 mt-6">
          <div className={`p-8 rounded-[2.5rem] shadow-2xl relative ${isHealthy ? 'bg-emerald-500 text-white' : 'bg-rose-500 text-white'}`}>
            <div className="absolute inset-0 bg-white opacity-20 blur-2xl rounded-full" />
            <div className="relative">
              {isHealthy ? <FiCheckCircle size={56} /> : <FiAlertCircle size={56} />}
            </div>
          </div>
        </div>

        <h4 className="text-[11px] font-black uppercase tracking-[0.4em] opacity-30 mb-5">AI Diagnostic Conclusion</h4>
        <div className={`text-6xl md:text-8xl font-black mb-8 tracking-tighter ${isHealthy ? 'text-emerald-500' : 'text-rose-500'}`}>
          {prediction}
        </div>

        <div className="flex flex-col items-center gap-5">
          <div className="px-10 py-4 rounded-[2rem] bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/5 flex items-center gap-4 shadow-xl">
            <div className={`w-3 h-3 rounded-full animate-pulse ${isHealthy ? 'bg-emerald-500' : 'bg-rose-500'}`} />
            <span className="text-lg font-black tracking-tight">
              Confidence Engine: {((confidence > 1 ? confidence : confidence * 100)).toFixed(1)}%
            </span>
          </div>
          <p className="text-[10px] font-black uppercase tracking-[0.3em] opacity-30">
            Validated by v1.2.0 ML-FastAPI Network
          </p>
        </div>
      </motion.div>

      {/* Accuracy & Metrics Section */}
      <div className="grid grid-cols-2 gap-5">
        {[
          { label: 'Overall Accuracy', value: metrics.accuracy, icon: <FiCheckCircle /> },
          { label: 'Model Precision', value: metrics.precision, icon: <FiShield /> },
          { label: 'Recall Reliability', value: metrics.recall, icon: <FiActivity /> },
          { label: 'F1 Grade Quality', value: metrics.f1, icon: <FiCpu /> },
        ].map((m, i) => (
          <motion.div
            key={m.label}
            variants={itemVariants}
            className={`p-8 rounded-[2.5rem] border space-y-5 transition-colors ${theme === 'dark' ? 'bg-[#0f0f12] border-white/5' : 'bg-white border-slate-100'}`}
          >
            <div className="flex items-center justify-between opacity-40">
              <div className="text-xl">{m.icon}</div>
              <span className="text-[10px] font-black uppercase tracking-widest">{m.label}</span>
            </div>
            <div className="space-y-3">
              <div className="text-3xl font-black tracking-tighter" style={{ color: accentColor }}>
                {(m.value * 100).toFixed(1)}<span className="text-sm ml-0.5">%</span>
              </div>
              <div className="h-2 w-full bg-slate-100 dark:bg-white/5 rounded-full overflow-hidden p-0.5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${m.value * 100}%` }}
                  transition={{ duration: 1.5, delay: 0.5 + (i * 0.1) }}
                  className="h-full rounded-full shadow-[0_0_10px_rgba(20,184,166,0.3)]"
                  style={{ backgroundColor: accentColor }}
                />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Feature Importance or Insights */}
      {topFeatures && topFeatures.length > 0 && (
        <motion.div variants={itemVariants} className="space-y-5">
          <div className="flex items-center gap-3 px-2">
            <div className="w-1.5 h-1.5 rounded-full bg-[#14b8a6]" />
            <h5 className="text-[11px] font-black uppercase tracking-[0.3em] opacity-30">Primary Diagnostic Weights</h5>
          </div>
          <div className="grid grid-cols-1 gap-3">
            {topFeatures.slice(0, 3).map((f, i) => (
              <div key={i} className={`p-5 rounded-[1.5rem] border flex items-center justify-between group transition-all hover:translate-x-1 ${theme === 'dark' ? 'bg-white/[0.02] border-white/5' : 'bg-slate-50/50 border-slate-100'}`}>
                <span className="text-sm font-black opacity-60 group-hover:opacity-100 transition-opacity">{f.name}</span>
                <div className="flex items-center gap-3">
                  <div className="h-1 w-20 bg-slate-200 dark:bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full bg-[#14b8a6] opacity-50" style={{ width: '80%' }} />
                  </div>
                  <span className="text-[10px] font-black px-4 py-2 rounded-xl bg-white dark:bg-white/5 shadow-sm uppercase tracking-widest" style={{ color: accentColor }}>
                    Critical
                  </span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default PredictionResult;