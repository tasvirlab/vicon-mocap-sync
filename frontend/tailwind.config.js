/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/renderer/src/**/*.{js,ts,jsx,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        bg:               'rgb(var(--color-bg)             / <alpha-value>)',
        surface:          'rgb(var(--color-surface)        / <alpha-value>)',
        card:             'rgb(var(--color-card)           / <alpha-value>)',
        border:           'rgb(var(--color-border)         / <alpha-value>)',
        text:             'rgb(var(--color-text)           / <alpha-value>)',
        muted:            'rgb(var(--color-muted)          / <alpha-value>)',
        dim:              'rgb(var(--color-dim)            / <alpha-value>)',
        accent:           'rgb(var(--color-accent)         / <alpha-value>)',
        accent2:          'rgb(var(--color-accent2)        / <alpha-value>)',
        'accent-hover':   'rgb(var(--color-accent-hover)   / <alpha-value>)',
        'accent-pressed': 'rgb(var(--color-accent-pressed) / <alpha-value>)',
        'on-accent':      'rgb(var(--color-on-accent)      / <alpha-value>)',
        tint:             'rgb(var(--color-accent) / 0.13)',
        accent3:          '#8B5CF6',
        success:          'rgb(var(--color-success) / <alpha-value>)',
        warning:          'rgb(var(--color-warning) / <alpha-value>)',
        danger:           'rgb(var(--color-danger)  / <alpha-value>)',
        destructive:      'rgb(var(--color-danger)  / <alpha-value>)',
      },
      fontFamily: {
        sans: ['Geist Variable', 'Geist', 'Segoe UI', 'system-ui', 'sans-serif'],
        mono: ['Geist Mono Variable', 'Geist Mono', 'ui-monospace', 'monospace'],
      },
      borderRadius: { sm: '6px', DEFAULT: '10px', md: '8px', lg: '10px', xl: '10px', '2xl': '10px' },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in':    'fadeIn 0.25s ease-out',
        'slide-up':   'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn:  { from: { opacity: '0' },               to: { opacity: '1' } },
        slideUp: { from: { opacity: '0', transform: 'translateY(12px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
      },
    },
  },
  plugins: [],
}
