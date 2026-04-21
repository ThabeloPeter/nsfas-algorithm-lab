import Head from 'next/head';
import dynamic from 'next/dynamic';

const FaceExtractionTool = dynamic(() => import('@/preprod/components/FaceExtractionTool'), {
  ssr: false
});

export default function HomePage() {
  return (
    <>
      <Head>
        <title>NSFAS Biometrics Verification</title>
        <meta
          name="description"
          content="Biometrics verification platform for secure identity checks"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </Head>

      <main className="min-h-screen bg-white dark:bg-black text-gray-900 dark:text-white transition-colors">
        <section className="border-b border-gray-200 dark:border-white/10 bg-white/90 dark:bg-black/90 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 md:py-8">
            <div className="flex items-center gap-4">
              <div className="relative w-12 h-12 md:w-16 md:h-16 bg-white rounded-lg p-1.5 md:p-2 shadow-lg flex-shrink-0">
                <img
                  src="/images/logo.png"
                  alt="NSFAS"
                  className="w-full h-full object-contain"
                />
              </div>
              <div>
                <p className="text-xs md:text-sm uppercase tracking-[0.2em] text-gray-500 dark:text-white/50">
                  Biometrics Verification
                </p>
                <h1 className="text-xl md:text-3xl font-bold tracking-tight">
                  Government-ready identity assurance platform
                </h1>
                <p className="text-sm md:text-base text-gray-600 dark:text-white/60 mt-1 font-light">
                  Capture an ID document, extract the face, and confirm identity in one flow.
                </p>
              </div>
            </div>
          </div>
        </section>

        <FaceExtractionTool />
      </main>
    </>
  );
}
