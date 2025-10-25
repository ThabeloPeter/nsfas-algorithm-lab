'use client';

import { useTheme } from '@/lib/ThemeProvider';
import { Sun, Moon, Monitor } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';

export default function ThemeToggle() {
  const { theme, setTheme, activeTheme } = useTheme();
  const [showMenu, setShowMenu] = useState(false);

  const themes = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor },
  ];

  const currentTheme = themes.find(t => t.value === theme);
  const CurrentIcon = currentTheme?.icon || Monitor;

  return (
    <div className="relative">
      {/* Toggle Button */}
      <button
        onClick={() => setShowMenu(!showMenu)}
        className="flex items-center space-x-2 px-3 py-2 rounded-lg border border-gray-300 dark:border-white/20 bg-white dark:bg-white/5 hover:bg-gray-50 dark:hover:bg-white/10 transition-all text-gray-700 dark:text-white text-sm"
        aria-label="Toggle theme"
      >
        <CurrentIcon className="w-4 h-4" />
        <span className="hidden sm:inline">{currentTheme?.label}</span>
      </button>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {showMenu && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-40"
              onClick={() => setShowMenu(false)}
            />
            
            {/* Menu */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.15 }}
              className="absolute right-0 mt-2 w-48 rounded-xl border border-gray-200 dark:border-white/20 bg-white dark:bg-gray-900 shadow-lg overflow-hidden z-50"
            >
              {themes.map((themeOption) => {
                const Icon = themeOption.icon;
                const isActive = theme === themeOption.value;
                
                return (
                  <button
                    key={themeOption.value}
                    onClick={() => {
                      setTheme(themeOption.value);
                      setShowMenu(false);
                    }}
                    className={`w-full flex items-center space-x-3 px-4 py-3 text-left transition-colors ${
                      isActive
                        ? 'bg-blue-50 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/5'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="flex-1 text-sm font-medium">{themeOption.label}</span>
                    {isActive && (
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                );
              })}
              
              {/* Current Active Theme Indicator */}
              <div className="px-4 py-2 border-t border-gray-200 dark:border-white/10 bg-gray-50 dark:bg-white/5">
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Active: <span className="font-semibold capitalize">{activeTheme}</span>
                </p>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

