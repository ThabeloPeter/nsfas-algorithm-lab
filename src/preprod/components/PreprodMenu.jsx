'use client';

import Link from 'next/link';
import { Scan, ArrowLeft, TestTube2, Sparkles, Play } from 'lucide-react';
import { motion } from 'framer-motion';

const testTools = [
  {
    id: 'face-extraction',
    title: 'Face Extraction',
    description: 'Extract and analyze face photos from ID documents with precision',
    icon: Scan,
    href: '/preprod/face-extraction',
    status: 'active',
    tag: 'Active'
  },
  // Add more tools here as you build them
  {
    id: 'coming-soon-1',
    title: 'Coming Soon',
    description: 'More innovative algorithms and solutions in development',
    icon: TestTube2,
    href: '#',
    status: 'disabled',
    tag: 'Soon'
  }
];

export default function PreprodMenu() {
  return (
    <div className="min-h-screen bg-black text-white" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      
      {/* Header */}
      <div className="bg-black border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 md:py-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center space-x-3 md:space-x-6">
              {/* NSFAS Logo */}
              <div className="relative w-12 h-12 md:w-16 md:h-16 bg-white rounded-lg p-1.5 md:p-2 shadow-lg flex-shrink-0">
                <img
                  src="/images/logo.png"
                  alt="NSFAS"
                  className="w-full h-full object-contain"
                />
              </div>
              <div>
                <h1 className="text-lg md:text-2xl font-bold text-white tracking-tight">
                  Algorithm Lab
                </h1>
                <p className="text-xs md:text-sm text-white/50 font-light mt-0.5 hidden sm:block">
                  Showcasing Innovative Solutions
                </p>
              </div>
            </div>
            <Link 
              href="/"
              className="flex items-center space-x-1.5 md:space-x-2 px-3 md:px-4 py-2 text-white/70 hover:text-white hover:bg-white/5 rounded-lg transition-all text-xs md:text-sm border border-white/10 hover:border-white/20"
            >
              <ArrowLeft className="w-3.5 h-3.5 md:w-4 md:h-4" />
              <span className="hidden sm:inline">Back to App</span>
              <span className="sm:hidden">Back</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 md:py-12">
        
        {/* Info Banner */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 md:mb-12 p-4 md:p-6 bg-white/5 border border-white/10 rounded-xl backdrop-blur-sm"
        >
          <div className="flex items-start space-x-3 md:space-x-4">
            <Sparkles className="w-4 h-4 md:w-5 md:h-5 text-white/70 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-white font-semibold mb-1 text-sm md:text-base">
                Algorithm Showcase & Testing Lab
              </p>
              <p className="text-white/60 text-xs md:text-sm font-light">
                Explore and test innovative algorithms and solutions designed for NSFAS. Each tool demonstrates cutting-edge approaches to real-world challenges.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Section Header */}
        <div className="mb-6 md:mb-8">
          <h2 className="text-lg md:text-xl font-bold text-white mb-2">Available Algorithms</h2>
          <div className="h-0.5 w-16 md:w-20 bg-gradient-to-r from-white to-transparent"></div>
        </div>

        {/* Tools Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
          {testTools.map((tool, index) => {
            const Icon = tool.icon;
            const isDisabled = tool.status === 'disabled';
            
            const cardContent = (
              <div className="relative h-full flex flex-col">
                {/* Status Tag */}
                <div className="absolute -top-2 md:-top-3 -right-2 md:-right-3 z-10">
                  <div className={`px-2.5 md:px-3 py-0.5 md:py-1 rounded-full text-xs font-medium shadow-lg ${
                    isDisabled 
                      ? 'bg-white/10 text-white/40' 
                      : 'bg-white text-black'
                  }`}>
                    {tool.tag}
                  </div>
                </div>

                {/* Icon */}
                <div className={`w-12 h-12 md:w-14 md:h-14 rounded-xl flex items-center justify-center mb-3 md:mb-4 ${
                  isDisabled 
                    ? 'bg-white/5' 
                    : 'bg-white/10'
                }`}>
                  <Icon className={`w-6 h-6 md:w-7 md:h-7 ${
                    isDisabled 
                      ? 'text-white/30' 
                      : 'text-white'
                  }`} />
                </div>

                {/* Title */}
                <h3 className={`text-base md:text-lg font-semibold mb-2 ${
                  isDisabled ? 'text-white/40' : 'text-white'
                }`}>
                  {tool.title}
                </h3>

                {/* Description */}
                <p className={`text-xs md:text-sm mb-4 md:mb-6 flex-grow font-light ${
                  isDisabled ? 'text-white/30' : 'text-white/60'
                }`}>
                  {tool.description}
                </p>

                {/* Action */}
                {!isDisabled && (
                  <div className="flex items-center space-x-2 text-xs md:text-sm font-medium text-white group-hover:translate-x-1 transition-transform">
                    <span>Try Algorithm</span>
                    <Play className="w-3.5 h-3.5 md:w-4 md:h-4" />
                  </div>
                )}
                {isDisabled && (
                  <div className="text-xs md:text-sm text-white/30">
                    Coming Soon
                  </div>
                )}
              </div>
            );

            return (
              <motion.div
                key={tool.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                {isDisabled ? (
                  <div className="bg-white/5 rounded-xl md:rounded-2xl border border-white/10 p-4 md:p-6 h-full cursor-not-allowed">
                    {cardContent}
                  </div>
                ) : (
                  <Link href={tool.href}>
                    <div className="bg-white/5 rounded-xl md:rounded-2xl border border-white/10 hover:border-white/30 hover:bg-white/10 transition-all p-4 md:p-6 h-full cursor-pointer group">
                      {cardContent}
                    </div>
                  </Link>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Info Section */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-10 md:mt-16 bg-white/5 rounded-xl md:rounded-2xl border border-white/10 p-4 md:p-8"
        >
          <h3 className="text-base md:text-lg font-semibold text-white mb-3 md:mb-4">Lab Information</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 md:gap-4 text-xs md:text-sm">
            <div>
              <p className="text-white/50 mb-1">Purpose</p>
              <p className="text-white">Algorithm showcase & testing environment</p>
            </div>
            <div>
              <p className="text-white/50 mb-1">Status</p>
              <p className="text-white">Active development & demonstration</p>
            </div>
            <div>
              <p className="text-white/50 mb-1">Technology</p>
              <p className="text-white">Next.js + AI/ML APIs</p>
            </div>
            <div>
              <p className="text-white/50 mb-1">Access URL</p>
              <code className="text-white font-mono text-xs bg-white/10 px-2 py-1 rounded break-all">/preprod</code>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

