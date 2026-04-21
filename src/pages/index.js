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
          content="Secure identity verification with guided document capture and selfie matching"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </Head>

      <main className="relative min-h-screen overflow-hidden bg-[#f4f7fb] text-slate-900 dark:bg-[#05070b] dark:text-white transition-colors">
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute -top-32 left-1/2 h-96 w-96 -translate-x-1/2 rounded-full bg-sky-400/20 blur-3xl dark:bg-cyan-400/10" />
          <div className="absolute right-[-8rem] top-48 h-80 w-80 rounded-full bg-indigo-400/15 blur-3xl dark:bg-blue-500/10" />
          <div className="absolute left-[-6rem] bottom-[-6rem] h-80 w-80 rounded-full bg-emerald-400/10 blur-3xl" />
          <div
            className="absolute inset-0 opacity-[0.16] dark:opacity-[0.09]"
            style={{
              backgroundImage:
                'linear-gradient(rgba(71,85,105,0.14) 1px, transparent 1px), linear-gradient(90deg, rgba(71,85,105,0.14) 1px, transparent 1px)',
              backgroundSize: '44px 44px'
            }}
          />
        </div>

        <section className="relative border-b border-white/60 bg-white/70 backdrop-blur-xl dark:border-white/10 dark:bg-slate-950/50">
          <div className="mx-auto flex max-w-7xl flex-col gap-8 px-4 py-8 md:px-6 md:py-12 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-4 py-2 text-xs font-semibold tracking-[0.18em] text-slate-600 shadow-sm dark:border-white/10 dark:bg-white/5 dark:text-white/60">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                Identity verification workflow
              </div>
              <div className="flex items-start gap-4">
                <div className="relative hidden h-16 w-16 flex-shrink-0 rounded-2xl bg-white p-2 shadow-xl ring-1 ring-slate-200 md:block dark:bg-white/5 dark:ring-white/10">
                  <img
                    src="/images/logo.png"
                    alt="NSFAS"
                    className="h-full w-full object-contain"
                  />
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.28em] text-slate-500 dark:text-white/45">
                    Biometrics Verification
                  </p>
                  <h1 className="mt-3 max-w-2xl text-3xl font-extrabold tracking-tight text-slate-950 sm:text-4xl md:text-5xl dark:text-white">
                    A refined capture and verification experience built for trust.
                  </h1>
                  <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600 sm:text-lg dark:text-white/65">
                    Capture a document, extract the face, and compare it with a live selfie in a calm, guided flow that feels fast, clear, and professional.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 lg:w-[460px]">
              {[
                { label: 'Capture', value: 'Guided' },
                { label: 'Extraction', value: 'Automated' },
                { label: 'Verification', value: 'Selfie match' }
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-[0_10px_30px_rgba(15,23,42,0.06)] backdrop-blur-xl dark:border-white/10 dark:bg-white/5"
                >
                  <p className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500 dark:text-white/45">
                    {item.label}
                  </p>
                  <p className="mt-2 text-sm font-semibold text-slate-900 dark:text-white">
                    {item.value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="relative mx-auto max-w-7xl px-4 py-6 md:px-6 md:py-10">
          <div className="mb-6 grid gap-4 md:grid-cols-3">
            {[
              {
                title: 'Document capture',
                text: 'A guided camera experience helps users place the ID correctly and stay aligned.'
              },
              {
                title: 'Face extraction',
                text: 'The best face is selected from the source document and returned with rich metadata.'
              },
              {
                title: 'Selfie verification',
                text: 'A second capture step compares the live face against the extracted ID image.'
              }
            ].map((card) => (
              <div
                key={card.title}
                className="rounded-3xl border border-slate-200/80 bg-white/85 p-5 shadow-[0_20px_50px_rgba(15,23,42,0.08)] backdrop-blur-xl dark:border-white/10 dark:bg-white/5"
              >
                <p className="text-sm font-semibold text-slate-900 dark:text-white">
                  {card.title}
                </p>
                <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-white/60">
                  {card.text}
                </p>
              </div>
            ))}
          </div>

          <div className="overflow-hidden rounded-[2rem] border border-slate-200/80 bg-white/80 shadow-[0_30px_80px_rgba(15,23,42,0.12)] backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
            <FaceExtractionTool />
          </div>
        </section>
      </main>
    </>
  );
}
