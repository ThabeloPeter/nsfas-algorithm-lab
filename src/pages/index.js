import { useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to Algorithm Lab
    router.push('/preprod');
  }, [router]);

  return (
    <>
      <Head>
        <title>NSFAS Algorithm Lab</title>
        <meta name="description" content="Algorithm showcase and testing environment for NSFAS" />
      </Head>
      <div className="min-h-screen bg-black text-white flex items-center justify-center" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">NSFAS Algorithm Lab</h1>
          <p className="text-white/60 mb-8">Redirecting to Algorithm Lab...</p>
          <div className="animate-pulse">
            <div className="w-2 h-2 bg-white rounded-full mx-auto"></div>
          </div>
        </div>
      </div>
    </>
  );
}

