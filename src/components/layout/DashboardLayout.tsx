import React, { ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTheme } from '../../context/ThemeContext';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { FiSun, FiMoon, FiActivity, FiHeart, FiDroplet, FiThermometer, FiSun as FiSkin, FiCrosshair, FiGrid } from 'react-icons/fi';
import { RiVirusFill } from 'react-icons/ri';

interface DashboardLayoutProps {
    children: ReactNode;
}

const navLinks = [
    { name: 'Dashboard', path: '/home', icon: <FiGrid /> },
    { name: 'Symptom Check', path: '/symptom', icon: <RiVirusFill /> },
    { name: 'Heart', path: '/heart', icon: <FiHeart /> },
    { name: 'Diabetes', path: '/diabetes', icon: <FiThermometer /> },
    { name: 'Liver', path: '/liver', icon: <FiDroplet /> },
    { name: 'Skin', path: '/skin', icon: <FiSkin /> },
    { name: 'Breast', path: '/breast', icon: <FiCrosshair /> },
    { name: 'Details', path: '/details', icon: <FiActivity /> },
];

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
    const { theme, toggleTheme, accentColor } = useTheme();
    const router = useRouter();

    return (
        <div
            className="min-h-screen transition-all duration-500 flex flex-col font-body overflow-x-hidden"
            style={{
                backgroundColor: theme === 'dark' ? '#050507' : '#f8fafc',
                color: theme === 'dark' ? '#ffffff' : '#1e1e24'
            }}
        >
            {/* Liquid Mesh Background */}
            <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
                <motion.div
                    className="absolute top-[-10%] left-[-10%] w-[60vh] h-[60vh] rounded-full mix-blend-multiply filter blur-[100px] opacity-[0.1]"
                    style={{ backgroundColor: accentColor }}
                    animate={{ x: [0, 100, -50, 0], y: [0, 50, 100, 0] }}
                    transition={{ duration: 20, repeat: Infinity }}
                />
                <motion.div
                    className="absolute bottom-[-10%] right-[-10%] w-[50vh] h-[50vh] rounded-full mix-blend-multiply filter blur-[100px] opacity-[0.15]"
                    style={{ backgroundColor: '#0ea5e9' }}
                    animate={{ x: [0, -80, 40, 0], y: [0, -100, -50, 0] }}
                    transition={{ duration: 18, repeat: Infinity }}
                />
                <div className="absolute inset-0 opacity-[0.03]"
                    style={{ backgroundImage: `radial-gradient(${theme === 'dark' ? '#fff' : '#000'} 1px, transparent 1px)`, backgroundSize: '40px 40px' }} />
            </div>

            {/* Top Header */}
            <header className="sticky top-0 z-50 px-8 py-4 flex justify-between items-center bg-white/40 dark:bg-black/20 backdrop-blur-xl border-b border-white/10">
                {/* Branding / Logo */}
                <Link href="/home" className="flex items-center gap-3 group">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#14b8a6] to-[#0ea5e9] flex items-center justify-center shadow-lg transform rotate-3 transition-transform group-hover:rotate-12">
                        <span className="text-white font-black text-xl">P</span>
                    </div>
                    <span className={`text-2xl font-black tracking-tighter ${theme === 'dark' ? 'text-white' : 'text-slate-800'}`}>
                        Predict<span className="text-[#14b8a6]">Well</span>
                    </span>
                </Link>

                {/* Central Navigation */}
                <nav className="hidden xl:flex items-center gap-1">
                    {navLinks.map((link) => {
                        const isActive = router.pathname === link.path;
                        return (
                            <Link key={link.path} href={link.path}>
                                <motion.div
                                    className={`px-4 py-2 rounded-xl flex items-center gap-2 text-sm font-bold transition-all cursor-pointer ${isActive
                                        ? 'bg-gradient-to-r from-[#14b8a6]/10 to-[#0ea5e9]/10 text-[#14b8a6]'
                                        : 'opacity-60 hover:opacity-100 hover:bg-slate-100 dark:hover:bg-white/5'
                                        }`}
                                >
                                    <span style={{ color: isActive ? accentColor : 'inherit' }}>{link.icon}</span>
                                    {link.name}
                                </motion.div>
                            </Link>
                        );
                    })}
                </nav>

                <div className="flex items-center gap-4">
                    {/* Theme toggle removed for light mode only theme */}
                </div>
            </header>

            {/* Mobile Navigation (Sub-header for links on small screens) */}
            <nav className="xl:hidden flex items-center justify-center gap-1 p-2 bg-slate-50/50 dark:bg-black/20 backdrop-blur-md overflow-x-auto no-scrollbar border-b border-white/5">
                {navLinks.map((link) => {
                    const isActive = router.pathname === link.path;
                    return (
                        <Link key={link.path} href={link.path}>
                            <motion.div
                                className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 text-[10px] font-black uppercase tracking-tighter transition-all cursor-pointer whitespace-nowrap ${isActive
                                    ? 'bg-[#14b8a6]/10 text-[#14b8a6]'
                                    : 'opacity-40 hover:opacity-100'
                                    }`}
                            >
                                {link.icon}
                                {link.name}
                            </motion.div>
                        </Link>
                    );
                })}
            </nav>

            <main className="flex-1 relative z-10 w-full">
                {children}
            </main>
        </div>
    );
};

export default DashboardLayout;
