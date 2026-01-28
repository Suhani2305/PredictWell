import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { gsap } from 'gsap';
import { TextPlugin } from 'gsap/dist/TextPlugin';

// Register GSAP plugins
if (typeof window !== 'undefined') {
  gsap.registerPlugin(TextPlugin);
}

interface TitleSectionProps {
  accentColor: string;
  theme: string;
  title?: string;
  subtitlePrefix?: string;
  subtitles?: string[];
  align?: 'left' | 'center';
}

const TitleSection: React.FC<TitleSectionProps> = ({
  accentColor,
  theme,
  title = "MedX - AI Medical Assistant",
  subtitlePrefix = "Quick Check",
  subtitles = [
    'Check your health risks instantly',
    'Get simple AI health reports',
    'Stay healthy with smart tips'
  ],
  align = 'center'
}) => {
  const textRef = useRef<HTMLDivElement>(null);

  // Text typing animation with GSAP
  useEffect(() => {
    if (typeof window === 'undefined' || !textRef.current) return;

    let currentIndex = 0;
    let delayedCall: gsap.core.Tween | null = null;

    const animateText = () => {
      if (!textRef.current) return;

      gsap.to(textRef.current, {
        duration: 0.5,
        text: { value: '', padSpace: true },
        ease: 'none',
        onComplete: () => {
          if (!textRef.current) return;
          currentIndex = (currentIndex + 1) % subtitles.length;
          gsap.to(textRef.current, {
            duration: 1.5,
            text: { value: subtitles[currentIndex], padSpace: true },
            ease: 'none',
            onComplete: () => {
              if (!textRef.current) return;
              delayedCall = gsap.delayedCall(3, animateText);
            }
          });
        }
      });
    };

    gsap.to(textRef.current, {
      duration: 1.5,
      text: { value: subtitles[0], padSpace: true },
      ease: 'none',
      onComplete: () => {
        if (!textRef.current) return;
        delayedCall = gsap.delayedCall(3, animateText);
      }
    });

    return () => {
      gsap.killTweensOf(textRef.current);
      if (delayedCall) delayedCall.kill();
    };
  }, [subtitles]);

  return (
    <div className={`space-y-4 ${align === 'center' ? 'text-center' : 'text-left'}`}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <h2 className={`text-5xl md:text-7xl font-black tracking-tighter leading-[1.1] pb-2 ${theme === 'dark' ? 'text-white' : 'text-[#050507]'}`}>
          {title}
        </h2>
      </motion.div>

      {/* Animated Typing Subtitle */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className={`flex items-center text-xl md:text-2xl font-medium tracking-tight ${align === 'center' ? 'justify-center' : 'justify-start'}`}
        style={{ color: theme === 'dark' ? '#94a3b8' : '#64748b' }}
      >
        <span className="mr-2 opacity-60">
          {subtitlePrefix}
        </span>
        <div
          ref={textRef}
          className="font-bold bg-clip-text text-transparent"
          style={{
            backgroundImage: `linear-gradient(to right, ${accentColor}, #0ea5e9)`,
          }}
        >
          {subtitles[0]}
        </div>
      </motion.div>

      {/* Underline decorative element */}
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: 100 }}
        transition={{ delay: 0.8, duration: 1 }}
        className={`h-1.5 rounded-full mx-auto ${align === 'center' ? 'mx-auto' : 'ml-0'}`}
        style={{ background: `linear-gradient(90deg, ${accentColor}, transparent)` }}
      />
    </div>
  );
};

export default TitleSection;
