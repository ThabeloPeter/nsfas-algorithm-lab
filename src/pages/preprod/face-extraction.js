import Head from 'next/head';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowLeft, Loader2 } from 'lucide-react';

// Dynamically import the component with no SSR to avoid face-api.js issues
const FaceExtractionTool = dynamic(() => import('@/preprod/components/FaceExtractionTool'), { 
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-black flex items-center justify-center" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      <div className="text-center">
        <Loader2 className="w-12 h-12 text-white animate-spin mx-auto mb-4" />
        <p className="text-white/60 text-sm font-light">Loading...</p>
      </div>
    </div>
  )
});

export default function FaceExtractionPage() {
  return (
    <>
      <Head>
        <title>Face Extraction - Testing Hub</title>
        <meta name="description" content="Test face extraction from ID documents" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      </Head>
      
      {/* Nav Bar */}
      <div className="bg-black border-b border-white/10 sticky top-0 z-10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-3">
          <Link 
            href="/preprod"
            className="inline-flex items-center space-x-2 px-4 py-2 text-white/70 hover:text-white hover:bg-white/5 rounded-lg transition-all text-sm border border-white/10 hover:border-white/20"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Hub</span>
          </Link>
        </div>
      </div>

      <FaceExtractionTool />
    </>
  );
}

