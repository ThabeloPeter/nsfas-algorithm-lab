import '@/styles/globals.css'
import { ThemeProvider } from '@/lib/ThemeProvider'

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider>
      <Component {...pageProps} />
    </ThemeProvider>
  )
}

