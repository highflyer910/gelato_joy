import { Fredoka } from 'next/font/google'
import ThemeRegistry from './ThemeRegistry'

const fredoka = Fredoka({ subsets: ['latin'] })


export const metadata = {
  title: 'Gelato Joy',
  description: 'Joy in every bite, bliss in every scoop!',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={fredoka.className}>
        <ThemeRegistry>
          {children}
        </ThemeRegistry>
      </body>
    </html>
  )
}