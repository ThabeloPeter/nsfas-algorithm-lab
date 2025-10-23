import Head from 'next/head';
import PreprodMenu from '@/preprod/components/PreprodMenu';

export default function PreprodPage() {
  return (
    <>
      <Head>
        <title>Algorithm Lab - NSFAS</title>
        <meta name="description" content="Showcasing innovative algorithms and solutions for NSFAS" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      </Head>
      
      <PreprodMenu />
    </>
  );
}


