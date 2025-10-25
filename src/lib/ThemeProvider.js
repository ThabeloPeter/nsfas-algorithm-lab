'use client';

import { createContext, useContext, useEffect, useState } from 'react';

const ThemeContext = createContext({
  theme: 'dark',
  setTheme: () => {},
  systemTheme: 'dark',
});

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('system'); // 'light', 'dark', or 'system'
  const [systemTheme, setSystemTheme] = useState('dark');
  const [mounted, setMounted] = useState(false);

  // Detect system theme preference
  useEffect(() => {
    setMounted(true);
    
    // Get stored preference or default to 'system'
    const storedTheme = localStorage.getItem('theme') || 'system';
    setTheme(storedTheme);

    // Detect system preference
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    setSystemTheme(mediaQuery.matches ? 'dark' : 'light');

    // Listen for system theme changes
    const handleChange = (e) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Apply theme to document
  useEffect(() => {
    if (!mounted) return;

    const root = document.documentElement;
    const activeTheme = theme === 'system' ? systemTheme : theme;

    if (activeTheme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme, systemTheme, mounted]);

  // Update theme and persist to localStorage
  const updateTheme = (newTheme) => {
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
  };

  // Prevent flash of unstyled content
  if (!mounted) {
    return <div style={{ visibility: 'hidden' }}>{children}</div>;
  }

  const activeTheme = theme === 'system' ? systemTheme : theme;

  return (
    <ThemeContext.Provider value={{ theme, setTheme: updateTheme, systemTheme, activeTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

