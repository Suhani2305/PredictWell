import React, { useRef, useState } from 'react';
import { motion, useMotionTemplate, useMotionValue } from 'framer-motion';
import { useTheme } from '../../context/ThemeContext';

interface GlassCardProps {
    children: React.ReactNode;
    className?: string;
    gradient?: string;
    hoverEffect?: boolean;
}

const GlassCard: React.FC<GlassCardProps> = ({
    children,
    className = '',
    gradient,
    hoverEffect = false
}) => {
    const { theme, accentColor } = useTheme();

    // Mouse tracking logic for spotlight effect
    const mouseX = useMotionValue(0);
    const mouseY = useMotionValue(0);

    function handleMouseMove({ currentTarget, clientX, clientY }: React.MouseEvent) {
        const { left, top } = currentTarget.getBoundingClientRect();
        mouseX.set(clientX - left);
        mouseY.set(clientY - top);
    }

    return (
        <motion.div
            whileHover={hoverEffect ? { y: -8, scale: 1.005 } : {}}
            onMouseMove={handleMouseMove}
            className={`group relative rounded-3xl overflow-hidden backdrop-blur-3xl transition-all duration-500 ease-out border ${className}`}
            style={{
                backgroundColor: theme === 'dark' ? 'rgba(20, 20, 25, 0.4)' : 'rgba(255, 255, 255, 0.7)',
                borderColor: theme === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.5)',
                boxShadow: theme === 'dark'
                    ? '0 8px 32px 0 rgba(0, 0, 0, 0.4)'
                    : '0 8px 32px 0 rgba(100, 116, 139, 0.08)'
            }}
        >
            {/* Spotlight Effect Gradient */}
            <motion.div
                className="pointer-events-none absolute -inset-px rounded-3xl opacity-0 transition duration-300 group-hover:opacity-100"
                style={{
                    background: useMotionTemplate`
            radial-gradient(
              500px circle at ${mouseX}px ${mouseY}px,
              ${accentColor}20,
              transparent 80%
            )
          ` as any
                }}
            />

            {/* Optional Fixed Gradient Overlay */}
            {gradient && (
                <div
                    className={`absolute inset-0 opacity-10 bg-gradient-to-br ${gradient}`}
                />
            )}

            {/* Content */}
            <div className="relative z-10 p-6 h-full flex flex-col">
                {children}
            </div>

            {/* Border Highlight on Hover */}
            <div
                className="absolute inset-x-0 bottom-0 h-[2px] w-full origin-left scale-x-0 transform opacity-0 transition-all duration-500 group-hover:scale-x-100 group-hover:opacity-100"
                style={{ background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)` }}
            />
        </motion.div>
    );
};

export default GlassCard;
