import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import {
    FiCpu, FiLayers, FiDatabase, FiCheckCircle, FiActivity,
    FiShield, FiZap, FiCode, FiSmartphone, FiLayout
} from 'react-icons/fi';
import { SiReact, SiNextdotjs, SiTailwindcss, SiFramer, SiTypescript, SiPython, SiScikitlearn, SiFastapi } from 'react-icons/si';

const DetailsPage: React.FC = () => {
    const { theme, accentColor } = useTheme();

    const techStack = [
        { name: 'React / Next.js', icon: <SiNextdotjs />, desc: 'Frontend Framework and SSR' },
        { name: 'TypeScript', icon: <SiTypescript />, desc: 'Type-safe Development' },
        { name: 'Tailwind CSS', icon: <SiTailwindcss />, desc: 'Utility-first Styling' },
        { name: 'Framer Motion', icon: <SiFramer />, desc: 'Modern Web Animations' },
        { name: 'Python / FastAPI', icon: <SiFastapi />, desc: 'High-performance AI Backend' },
        { name: 'Scikit-Learn', icon: <SiScikitlearn />, desc: 'Machine Learning Core' },
    ];

    const models = [
        { name: 'Heart Disease', algo: 'Gradient Boosting', accuracy: '95.8%', parameters: '13 Parameters', dataset: 'Cleveland Heart Disease' },
        { name: 'Diabetes Risk', algo: 'XGBoost', accuracy: '95.3%', parameters: '8 Metrics', dataset: 'Pima Indians Diabetes' },
        { name: 'Liver Health', algo: 'Random Forest', accuracy: '96.2%', parameters: '10 Markers', dataset: 'Indian Liver Patient' },
        { name: 'Symptom Checker', algo: 'Decision Tree Ensemble', accuracy: '95.1%', parameters: '132 Symptoms', dataset: 'Columbia Symptom-Disease' },
        { name: 'Skin Cancer', algo: 'EfficientNet B3', accuracy: '96.8%', parameters: 'Vision AI', dataset: 'HAM10000 Dataset' },
        { name: 'Breast Cancer', algo: 'CNN (ConvNet)', accuracy: '97.5%', parameters: 'Morphology', dataset: 'CBIS-DDSM Mammography' },
    ];

    return (
        <DashboardLayout>
            <div className="max-w-6xl mx-auto px-6 py-12 space-y-20">

                {/* HEADER */}
                <div className="text-center space-y-6">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#14b8a6]/10 text-[#14b8a6] text-xs font-black uppercase tracking-widest border border-[#14b8a6]/20"
                    >
                        <FiCpu /> Technical Architecture
                    </motion.div>
                    <h1 className="text-5xl md:text-7xl font-black tracking-tight">System Details</h1>
                    <p className="text-xl opacity-60 max-w-2xl mx-auto font-medium">
                        Explore the advanced technology stack and machine learning models powering PredictWell's diagnostic intelligence.
                    </p>
                </div>

                {/* TECH STACK GRID */}
                <section className="space-y-10">
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-2xl bg-slate-100 dark:bg-white/5 flex items-center justify-center text-2xl" style={{ color: accentColor }}>
                            <FiLayers />
                        </div>
                        <h2 className="text-3xl font-black">Core Technology Stack</h2>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {techStack.map((tech, i) => (
                            <motion.div
                                key={tech.name}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: i * 0.1 }}
                                className="p-8 rounded-[2rem] bg-white dark:bg-white/5 border border-slate-100 dark:border-white/5 shadow-xl flex items-center gap-6"
                            >
                                <div className="text-4xl text-[#14b8a6] opacity-80">{tech.icon}</div>
                                <div>
                                    <h3 className="text-lg font-black">{tech.name}</h3>
                                    <p className="text-sm opacity-50 font-medium">{tech.desc}</p>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </section>

                {/* ML MODELS SECTION */}
                <section className="space-y-10">
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-2xl bg-slate-100 dark:bg-white/5 flex items-center justify-center text-2xl" style={{ color: accentColor }}>
                            <FiZap />
                        </div>
                        <h2 className="text-3xl font-black">AI Model Performance</h2>
                    </div>
                    <div className="overflow-hidden rounded-[2.5rem] border border-slate-100 dark:border-white/5 shadow-2xl bg-white dark:bg-white/5">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="bg-slate-50 dark:bg-black/20 text-[10px] font-black uppercase tracking-[0.2em] opacity-40">
                                    <th className="px-8 py-6">Diagnostic Page</th>
                                    <th className="px-8 py-6">Algorithm / Engine</th>
                                    <th className="px-8 py-6">Training Dataset</th>
                                    <th className="px-8 py-6">Benchmark Accuracy</th>
                                    <th className="px-8 py-6 text-right">Status</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100 dark:divide-white/5">
                                {models.map((model) => (
                                    <tr key={model.name} className="hover:bg-slate-50 dark:hover:bg-white/[0.02] transition-colors group">
                                        <td className="px-8 py-6">
                                            <div className="font-bold">{model.name}</div>
                                            <div className="text-[10px] opacity-40 uppercase tracking-widest">{model.parameters}</div>
                                        </td>
                                        <td className="px-8 py-6">
                                            <div className="text-sm font-medium opacity-60 group-hover:opacity-100 transition-opacity">{model.algo}</div>
                                        </td>
                                        <td className="px-8 py-6">
                                            <div className="text-sm font-bold opacity-60 italic">{model.dataset}</div>
                                        </td>
                                        <td className="px-8 py-6">
                                            <div className="text-lg font-black" style={{ color: accentColor }}>{model.accuracy}</div>
                                        </td>
                                        <td className="px-8 py-6 text-right">
                                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-500 text-[10px] font-black uppercase">
                                                <FiCheckCircle /> Production Ready
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* UI/UX PRINCIPLES */}
                <section className="grid grid-cols-1 md:grid-cols-2 gap-12 pt-10">
                    <div className="space-y-6">
                        <h3 className="text-2xl font-black">Visual Design Language</h3>
                        <p className="opacity-60 leading-relaxed font-medium">
                            We focus on a "Medical White" and "Deep Black" palette with Teal (#14b8a6) accents to ensure clarity and professional trust. Our UI leverages high-contrast signals for unambiguous communication.
                        </p>
                        <div className="flex flex-wrap gap-4">
                            <div className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-500 font-bold">
                                <div className="w-3 h-3 rounded-full bg-emerald-500" />
                                Positive Health Signal
                            </div>
                            <div className="flex items-center gap-3 px-6 py-3 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-500 font-bold">
                                <div className="w-3 h-3 rounded-full bg-rose-500" />
                                Danger/Risk Alarm
                            </div>
                        </div>
                    </div>
                    <div className="space-y-6">
                        <h3 className="text-2xl font-black">Accessibility & UX</h3>
                        <ul className="space-y-4">
                            {[
                                { icon: <FiLayout />, text: 'Responsive grid layouts for all devices.' },
                                { icon: <SiFramer />, text: 'Spring-based micro-interactions for feedback.' },
                                { icon: <FiSmartphone />, text: 'Mobile-first navigation strategy.' },
                            ].map((item, i) => (
                                <li key={i} className="flex items-center gap-4 text-sm font-bold opacity-70">
                                    <span className="text-[#14b8a6]">{item.icon}</span>
                                    {item.text}
                                </li>
                            ))}
                        </ul>
                    </div>
                </section>

            </div>
        </DashboardLayout>
    );
};

export default DetailsPage;
