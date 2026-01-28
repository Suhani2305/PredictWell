import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import GlassCard from '../components/ui/GlassCard';
import { useRouter } from 'next/router';
import {
  FiHeart, FiDroplet, FiThermometer, FiSun, FiInfo,
  FiCrosshair, FiArrowRight, FiActivity, FiTrendingUp, FiShield, FiUsers, FiClock
} from 'react-icons/fi';
import { RiVirusFill, RiMentalHealthLine } from 'react-icons/ri';

// Define model card data
const modelCards = [
  {
    id: 'symptom',
    title: 'Symptom Check',
    description: 'AI-based analysis of your symptoms to suggest potential conditions.',
    route: '/symptom',
    icon: <RiVirusFill size={28} />,
    color: '#14b8a6', // Teal
    stats: '95% Accuracy',
    image: '/about.webp'
  },
  {
    id: 'heart',
    title: 'Heart Health',
    description: 'Cardiovascular risk assessment using advanced ML algorithms.',
    route: '/heart',
    icon: <FiHeart size={28} />,
    color: '#ef4444', // Red
    stats: '14 Parameters',
    image: '/heart.png'
  },
  {
    id: 'diabetes',
    title: 'Diabetes Risk',
    description: 'Early detection of diabetes based on glucose, BMI, and age factors.',
    route: '/diabetes',
    icon: <FiThermometer size={28} />,
    color: '#f59e0b', // Amber
    stats: 'Type 1 & 2',
    image: '/diabetes.png'
  },
  {
    id: 'liver',
    title: 'Liver Function',
    description: 'Hepatic health evaluation using biological markers.',
    route: '/liver',
    icon: <FiDroplet size={28} />,
    color: '#0ea5e9', // Sky Blue
    stats: 'Enzyme Analysis',
    image: '/liver.png'
  },
  {
    id: 'skin',
    title: 'Skin Analysis',
    description: 'Dermatological screening for skin lesions and abnormalities.',
    route: '/skin',
    icon: <FiSun size={28} />,
    color: '#f97316', // Orange
    stats: 'Vision AI',
    image: '/skin.png'
  },
  {
    id: 'breast',
    title: 'Breast Cancer',
    description: 'Oncology screening assistant for breast tissue analysis.',
    route: '/breast',
    icon: <FiCrosshair size={28} />,
    color: '#ec4899', // Pink
    stats: 'Cell Analysis',
    image: '/breast.png'
  },
];

const HomePage: React.FC = () => {
  const { theme, accentColor } = useTheme();
  const router = useRouter();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.3 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 70, damping: 15 }
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-[1400px] mx-auto overflow-hidden">

        {/* HERO SECTION */}
        <div className="relative pt-12 pb-20 px-4 md:px-8 flex flex-col lg:flex-row items-center gap-12">
          <div className="flex-1 space-y-8 z-10 text-center lg:text-left">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className={`text-5xl md:text-7xl font-black tracking-tight leading-[1.1] ${theme === 'dark' ? 'text-white' : 'text-[#050507]'}`}>
                Check Your Health <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#14b8a6] to-[#0ea5e9]">
                  In Seconds.
                </span>
              </h1>
            </motion.div>

            <motion.p
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className={`text-xl md:text-2xl font-medium leading-relaxed max-w-2xl mx-auto lg:mx-0 ${theme === 'dark' ? 'text-slate-400' : 'text-[#050507]/60'}`}
            >
              Take control of your health with our simple AI tools. Find out early if you are at risk for common diseases and stay healthy.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start"
            >
              <button
                onClick={() => document.getElementById('models')?.scrollIntoView({ behavior: 'smooth' })}
                className="px-10 py-5 rounded-2xl font-bold text-white shadow-2xl transition-all hover:scale-105 active:scale-95"
                style={{
                  background: `linear-gradient(135deg, ${accentColor}, #0ea5e9)`,
                  boxShadow: `0 20px 40px -10px ${accentColor}60`
                }}
              >
                Start Free Check
              </button>
              <button className={`px-10 py-5 rounded-2xl font-bold border-2 transition-all active:scale-95 ${theme === 'dark' ? 'bg-white/5 border-white/10 text-white' : 'bg-white border-[#050507]/10 text-[#050507]'}`}>
                How it works
              </button>
            </motion.div>
          </div>

          {/* Hero Image Section - Clean Floating Heart */}
          <div className="flex-1 relative flex items-center justify-center py-20">
            <motion.div
              style={{
                perspective: 1000,
              }}
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width - 0.5;
                const y = (e.clientY - rect.top) / rect.height - 0.5;
                e.currentTarget.style.setProperty('--x', `${x * 40}deg`);
                e.currentTarget.style.setProperty('--y', `${-y * 40}deg`);
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.setProperty('--x', '0deg');
                e.currentTarget.style.setProperty('--y', '0deg');
              }}
              className="relative w-full max-w-[500px] transition-transform duration-200 ease-out"
              animate={{
                y: [0, -20, 0],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <motion.img
                src="/heart.png"
                alt="Heart Health"
                className="w-full h-auto drop-shadow-[0_50px_50px_rgba(239,68,68,0.3)] transition-all duration-300"
                style={{
                  transform: 'rotateX(var(--y, 0deg)) rotateY(var(--x, 0deg))',
                } as any}
              />

               

              {/* Dynamic Glow behind the heart */}
              <div
                className="absolute inset-0 blur-[120px] opacity-20 -z-10 rounded-full"
                style={{ background: `radial-gradient(circle, ${accentColor}, transparent)` }}
              />
            </motion.div>
          </div>
        </div>

        {/* FEATURES STRIP */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 px-4 md:px-8 mb-24">
          {[
            { title: 'Easy to Use', desc: 'Just enter your details and get reports.', icon: <FiActivity color="#14b8a6" /> },
            { title: 'Full Privacy', desc: 'Your data stays safe and secure with us.', icon: <FiShield color="#0ea5e9" /> },
            { title: 'Fast Results', desc: 'Get your health scan results instantly.', icon: <FiClock color="#6366f1" /> },
          ].map((f, i) => (
            <motion.div
              key={i}
              whileHover={{ y: -10 }}
              className={`p-8 rounded-[2rem] border shadow-xl flex flex-col items-center text-center space-y-4 ${theme === 'dark' ? 'bg-[#0f0f12] border-white/5 shadow-none' : 'bg-white border-[#050507]/5 shadow-slate-200/50'}`}
            >
              <div className="w-16 h-16 rounded-2xl bg-slate-50 dark:bg-white/5 flex items-center justify-center text-2xl shadow-inner">
                {f.icon}
              </div>
              <h3 className="text-xl font-bold">{f.title}</h3>
              <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* MODELS GRID SECTION */}
        <section id="models" className="px-4 md:px-8 pb-12">
          <div className="flex flex-col md:flex-row justify-between items-end mb-12 gap-6">
            <div>
              <h2 className="text-4xl md:text-5xl font-black mb-4 tracking-tight">Available Analysis Models</h2>
              <p className="text-slate-500 text-lg max-w-xl">Choose a specific health area to perform an AI-powered diagnostic check based on your data.</p>
            </div>
            <div className="flex gap-2">
              <div className="px-4 py-2 rounded-full bg-[#14b8a6]/10 text-[#14b8a6] text-xs font-bold uppercase tracking-widest border border-[#14b8a6]/20">
                {modelCards.length} Models Active
              </div>
            </div>
          </div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-10"
          >
            {modelCards.map((card) => (
              <motion.div
                key={card.id}
                variants={itemVariants}
                onClick={() => router.push(card.route)}
                className="group"
              >
                <GlassCard
                  hoverEffect={true}
                  className="h-full group cursor-pointer border-none !bg-white dark:!bg-white/5 shadow-2xl shadow-slate-200/60 dark:shadow-none !rounded-[2.5rem] p-0 overflow-hidden"
                >
                  <div className="relative h-48 overflow-hidden rounded-t-[2.5rem]">
                    <motion.img
                      whileHover={{ scale: 1.1, rotate: -2 }}
                      transition={{ duration: 0.6 }}
                      src={card.image}
                      alt={card.title}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                    <div className="absolute bottom-4 left-6 flex items-center gap-3">
                      <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center shadow-lg backdrop-blur-md border border-white/20"
                        style={{
                          backgroundColor: `${card.color}40`,
                          color: '#fff',
                        }}
                      >
                        {card.icon}
                      </div>
                      <span className="text-white font-black uppercase tracking-widest text-[10px]">{card.stats}</span>
                    </div>
                  </div>

                  <div className="p-8 space-y-4">
                    <div className="flex justify-between items-center">
                      <h3 className="text-2xl font-black tracking-tight group-hover:text-[#14b8a6] transition-colors">
                        {card.title}
                      </h3>
                      <div className="w-10 h-10 rounded-full border border-slate-200 dark:border-white/10 flex items-center justify-center text-slate-400 group-hover:text-white group-hover:bg-[#14b8a6] group-hover:border-[#14b8a6] transition-all duration-300">
                        <FiArrowRight />
                      </div>
                    </div>
                    <p className={`text-base leading-relaxed font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}`}>
                      {card.description}
                    </p>
                  </div>
                </GlassCard>
              </motion.div>
            ))}
          </motion.div>
        </section>

        {/* FOOTER CALLOUT */}
        <section className="mt-20 mb-12 px-4 md:px-8">
          <div className="rounded-[3rem] p-12 md:p-20 text-center relative overflow-hidden text-white"
            style={{ background: `linear-gradient(135deg, ${accentColor}, #0ea5e9)` }}>
            <div className="absolute top-0 left-0 w-full h-full opacity-10 pointer-events-none">
              <FiActivity className="absolute -top-10 -left-10 w-64 h-64 rotate-12" />
              <FiHeart className="absolute -bottom-10 -right-10 w-96 h-96 -rotate-12" />
            </div>
            <h2 className="text-4xl md:text-6xl font-black mb-8 relative z-10">Start your primary check today.</h2>
            <p className="text-xl md:text-2xl opacity-90 mb-10 max-w-3xl mx-auto relative z-10 leading-relaxed font-medium">
              Early detection is the key to preventing long-term health complications. Use our AI-powered models for a professional-grade screening.
            </p>
            <button className="px-12 py-6 rounded-2xl bg-white text-[#14b8a6] font-black text-xl shadow-2xl hover:scale-105 active:scale-95 transition-all relative z-10">
              Get Analysis Now
            </button>
          </div>
        </section>
      </div>
    </DashboardLayout>
  );
};

export default HomePage;
