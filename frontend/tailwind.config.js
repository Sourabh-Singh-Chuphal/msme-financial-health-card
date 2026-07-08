/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#F6F2EC',
        card: '#FDFCFA',
        'text-primary': '#17181C',
        'text-secondary': '#6B6A66',
        accent: {
          lavender: '#A99EF2',
          coral: '#F2916B',
          pink: '#F0A8C0',
        },
        success: {
          mint: '#4CAF7D',
        },
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Space Grotesk', 'sans-serif'],
      },
      boxShadow: {
        'subtle-card': '0 1px 2px rgba(23,24,28,0.04), 0 8px 24px rgba(23,24,28,0.06)',
      }
    },
  },
  plugins: [],
}
