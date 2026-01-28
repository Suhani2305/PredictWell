import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';

type Theme = 'light' | 'dark';

interface ColorPalette {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  border: string;
  error: string;
  success: string;
  warning: string;
  info: string;
}

interface ThemeContextType {
  theme: Theme;
  colors: ColorPalette;
  accentColor: string;
  toggleTheme: () => void;
  setAccentColor: (color: string) => void;
  getThemeColor: (colorName: keyof ColorPalette) => string;
}

// Predefined color palettes with improved aesthetics
const darkPalette: ColorPalette = {
  primary: '#050507',      // Deep Black
  secondary: '#0f0f12',    // Slightly lighter for layering
  accent: '#14b8a6',       // Teal
  background: '#050507',   // Deep Black
  surface: '#0f0f12',      // Deep Black Surface
  text: '#ffffff',         // White
  textSecondary: '#94a3b8', // Slate-400
  border: '#1e293b',       // Dark Border
  error: '#f43f5e',        // Rose Red (Dangerous)
  success: '#10b981',      // Emerald Green (Healthy)
  warning: '#f59e0b',      // Amber
  info: '#0ea5e9'          // Sky Blue
};

const lightPalette: ColorPalette = {
  primary: '#ffffff',      // Surgical White
  secondary: '#f8fafc',    // Surgical White subtle
  accent: '#14b8a6',       // Teal
  background: '#f8fafc',   // Surgical White background
  surface: '#ffffff',      // White surface
  text: '#0f172a',         // Deep Navy/Black text
  textSecondary: '#64748b', // Slate-500
  border: '#e2e8f0',       // Subtle border
  error: '#e11d48',        // Strong Red
  success: '#059669',      // Strong Green
  warning: '#d97706',      // Darker amber
  info: '#0284c7'          // Darker sky blue
};

// Generate a color palette based on accent color with harmonizing colors
const generatePalette = (baseTheme: ColorPalette, accentColor: string): ColorPalette => {
  // Convert hex to HSL for easier manipulation
  const hexToHSL = (hex: string): { h: number, s: number, l: number } => {
    // Remove the # if present
    hex = hex.replace(/^#/, '');

    // Parse the RGB values
    const r = parseInt(hex.substring(0, 2), 16) / 255;
    const g = parseInt(hex.substring(2, 4), 16) / 255;
    const b = parseInt(hex.substring(4, 6), 16) / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0, s = 0;
    const l = (max + min) / 2;

    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

      switch (max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
      }

      h /= 6;
    }

    return { h: h * 360, s: s * 100, l: l * 100 };
  };

  // Convert HSL back to hex
  const hslToHex = (h: number, s: number, l: number): string => {
    h /= 360;
    s /= 100;
    l /= 100;

    let r, g, b;

    if (s === 0) {
      r = g = b = l;
    } else {
      const hue2rgb = (p: number, q: number, t: number) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };

      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;

      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }

    const toHex = (x: number) => {
      const hex = Math.round(x * 255).toString(16);
      return hex.length === 1 ? '0' + hex : hex;
    };

    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  };

  // Get HSL values of the accent color
  const accentHSL = hexToHSL(accentColor);

  // Create complementary colors based on the accent
  const complementary = hslToHex((accentHSL.h + 180) % 360, accentHSL.s, accentHSL.l);
  const analogous1 = hslToHex((accentHSL.h + 30) % 360, accentHSL.s, accentHSL.l);
  const analogous2 = hslToHex((accentHSL.h - 30 + 360) % 360, accentHSL.s, accentHSL.l);

  // Adjust error, success, warning colors to harmonize with the accent
  return {
    ...baseTheme,
    accent: accentColor,
    info: analogous1,
    success: baseTheme === darkPalette ?
      hslToHex((accentHSL.h + 90) % 360, Math.min(accentHSL.s + 10, 100), Math.min(accentHSL.l + 5, 100)) :
      hslToHex((accentHSL.h + 90) % 360, Math.min(accentHSL.s + 5, 100), Math.max(accentHSL.l - 10, 20)),
    warning: baseTheme === darkPalette ?
      hslToHex((accentHSL.h + 150) % 360, Math.min(accentHSL.s, 100), Math.min(accentHSL.l + 10, 100)) :
      hslToHex((accentHSL.h + 150) % 360, Math.min(accentHSL.s, 100), Math.max(accentHSL.l - 5, 30)),
  };
};

const defaultAccentColor = '#14b8a6'; // Cyan-Teal as default

const defaultContext: ThemeContextType = {
  theme: 'light',
  colors: lightPalette,
  accentColor: defaultAccentColor,
  toggleTheme: () => { },
  setAccentColor: () => { },
  getThemeColor: () => '#000000',
};

const ThemeContext = createContext<ThemeContextType>(defaultContext);

export const useTheme = () => useContext(ThemeContext);

interface ThemeProviderProps {
  children: ReactNode;
}

// Fixed accent color
export const accentColors = ['#14b8a6'];

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('light');
  const [accentColor, setAccentColor] = useState<string>(defaultAccentColor);
  const [colors, setColors] = useState<ColorPalette>(
    generatePalette(lightPalette, accentColor)
  );

  useEffect(() => {
    // Force light theme
    setTheme('light');
    localStorage.setItem('theme', 'light');
  }, []);

  // Update colors when theme or accent color changes
  useEffect(() => {
    const basePalette = theme === 'dark' ? darkPalette : lightPalette;
    const newColors = generatePalette(basePalette, accentColor);
    setColors(newColors);

    // Apply theme to document body
    document.body.classList.toggle('dark-theme', theme === 'dark');
    document.body.classList.toggle('light-theme', theme === 'light');

    // Set CSS variables for colors
    Object.entries(newColors).forEach(([key, value]) => {
      document.documentElement.style.setProperty(`--color-${key}`, value);
    });

    // Set additional CSS variables for gradients and effects
    document.documentElement.style.setProperty('--accent-gradient',
      `linear-gradient(135deg, ${newColors.accent}, ${newColors.info})`);

    document.documentElement.style.setProperty('--accent-glow',
      `0 0 20px ${newColors.accent}80, 0 0 40px ${newColors.accent}40`);

    document.documentElement.style.setProperty('--surface-gradient',
      theme === 'dark' ?
        `linear-gradient(180deg, ${newColors.surface}, ${newColors.primary})` :
        `linear-gradient(180deg, ${newColors.primary}, ${newColors.surface})`);

    // Apply accent color to all elements with data-accent-color attribute
    document.querySelectorAll('[data-accent-color="true"]').forEach(element => {
      if (element instanceof HTMLElement) {
        element.style.backgroundColor = accentColor;
      }
    });

    // Apply gradient backgrounds with accent color
    document.querySelectorAll('[data-accent-gradient="true"]').forEach(element => {
      if (element instanceof HTMLElement) {
        element.style.background = `linear-gradient(135deg, ${accentColor}22 0%, ${theme === 'dark' ? '#000000' : '#ffffff'} 100%)`;
      }
    });
  }, [theme, accentColor]);

  const toggleTheme = () => {
    // Theme toggle disabled, always light
    setTheme('light');
    localStorage.setItem('theme', 'light');
  };

  const handleSetAccentColor = (color: string) => {
    // No longer allowing dynamic changes
    console.log("Accent color is fixed.");
  };

  const getThemeColor = (colorName: keyof ColorPalette): string => {
    return colors[colorName];
  };

  return (
    <ThemeContext.Provider
      value={{
        theme,
        colors,
        accentColor,
        toggleTheme,
        setAccentColor: handleSetAccentColor,
        getThemeColor,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};
