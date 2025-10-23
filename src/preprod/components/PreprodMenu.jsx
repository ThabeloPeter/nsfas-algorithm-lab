'use client';

import Link from 'next/link';
import Image from 'next/image';
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
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              {/* NSFAS Logo */}
              <div className="relative w-16 h-16 bg-white rounded-lg p-2 shadow-lg">
                <Image
                  src="/images/logo.png"
                  alt="NSFAS"
                  width={48}
                  height={48}
                  className="object-contain"
                />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">
                  Algorithm Lab
                </h1>
                <p className="text-sm text-white/50 font-light mt-0.5">
                  Showcasing Innovative Solutions
                </p>
              </div>
            </div>
            <Link 
              href="/"
              className="flex items-center space-x-2 px-4 py-2 text-white/70 hover:text-white hover:bg-white/5 rounded-lg transition-all text-sm border border-white/10 hover:border-white/20"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to App</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        
        {/* Info Banner */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12 p-6 bg-white/5 border border-white/10 rounded-xl backdrop-blur-sm"
        >
          <div className="flex items-start space-x-4">
            <Sparkles className="w-5 h-5 text-white/70 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-white font-semibold mb-1">
                Algorithm Showcase & Testing Lab
              </p>
              <p className="text-white/60 text-sm font-light">
                Explore and test innovative algorithms and solutions designed for NSFAS. Each tool demonstrates cutting-edge approaches to real-world challenges.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Section Header */}
        <div className="mb-8">
          <h2 className="text-xl font-bold text-white mb-2">Available Algorithms</h2>
          <div className="h-0.5 w-20 bg-gradient-to-r from-white to-transparent"></div>
        </div>

        {/* Tools Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {testTools.map((tool, index) => {
            const Icon = tool.icon;
            const isDisabled = tool.status === 'disabled';
            
            const cardContent = (
              <div className="relative h-full flex flex-col">
                {/* Status Tag */}
                <div className="absolute -top-3 -right-3 z-10">
                  <div className={`px-3 py-1 rounded-full text-xs font-medium shadow-lg ${
                    isDisabled 
                      ? 'bg-white/10 text-white/40' 
                      : 'bg-white text-black'
                  }`}>
                    {tool.tag}
                  </div>
                </div>

                {/* Icon */}
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center mb-4 ${
                  isDisabled 
                    ? 'bg-white/5' 
                    : 'bg-white/10'
                }`}>
                  <Icon className={`w-7 h-7 ${
                    isDisabled 
                      ? 'text-white/30' 
                      : 'text-white'
                  }`} />
                </div>

                {/* Title */}
                <h3 className={`text-lg font-semibold mb-2 ${
                  isDisabled ? 'text-white/40' : 'text-white'
                }`}>
                  {tool.title}
                </h3>

                {/* Description */}
                <p className={`text-sm mb-6 flex-grow font-light ${
                  isDisabled ? 'text-white/30' : 'text-white/60'
                }`}>
                  {tool.description}
                </p>

                {/* Action */}
                {!isDisabled && (
                  <div className="flex items-center space-x-2 text-sm font-medium text-white group-hover:translate-x-1 transition-transform">
                    <span>Try Algorithm</span>
                    <Play className="w-4 h-4" />
                  </div>
                )}
                {isDisabled && (
                  <div className="text-sm text-white/30">
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
                  <div className="bg-white/5 rounded-2xl border border-white/10 p-6 h-full cursor-not-allowed">
                    {cardContent}
                  </div>
                ) : (
                  <Link href={tool.href}>
                    <div className="bg-white/5 rounded-2xl border border-white/10 hover:border-white/30 hover:bg-white/10 transition-all p-6 h-full cursor-pointer group">
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
          className="mt-16 bg-white/5 rounded-2xl border border-white/10 p-8"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Lab Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
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
              <code className="text-white font-mono text-xs bg-white/10 px-2 py-1 rounded">/preprod</code>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

