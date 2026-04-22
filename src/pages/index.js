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
        <section className="border-b border-gray-200 dark:border-white/10">
          <div className="max-w-7xl mx-auto px-4 md:px-6 py-5 md:py-6">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 md:gap-4">
                <div className="relative w-10 h-10 md:w-12 md:h-12 bg-white rounded-lg p-1.5 shadow-sm flex-shrink-0 border border-gray-200">
                  <img
                    src="/images/logo.png"
                    alt="NSFAS"
                    className="w-full h-full object-contain"
                  />
                </div>
                <div>
                  <p className="text-[10px] md:text-xs uppercase tracking-[0.25em] text-gray-500 dark:text-white/45">
                    Biometrics Verification
                  </p>
                  <h1 className="text-lg md:text-2xl font-semibold tracking-tight">
                    Identity capture and verification
                  </h1>
                </div>
              </div>
              <p className="hidden md:block text-sm text-gray-600 dark:text-white/55 max-w-md text-right">
                Capture the document, isolate the face, and verify it with a clean workflow.
              </p>
            </div>
          </div>
        </section>

        <FaceExtractionTool />
      </main>
    </>
  );
}
