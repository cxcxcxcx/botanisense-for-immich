/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      // Palette tokens are CSS custom properties (rgb triples — Tailwind
      // combines them with the alpha channel via rgb(var(...) /
      // <alpha-value>)). Defined in app.css under `:root` (dark
      // defaults) and `[data-theme='light']` (light overrides). Switching
      // themes changes the variables, which propagates to every class
      // — including opacity variants like `bg-bg/90`, which used to be
      // hardcoded rgba() at build time and bypassed the theme.
      colors: {
        bg:      { DEFAULT: 'rgb(var(--c-bg) / <alpha-value>)', 0: 'rgb(var(--c-bg-0) / <alpha-value>)', 1: 'rgb(var(--c-bg-1) / <alpha-value>)', 2: 'rgb(var(--c-bg-2) / <alpha-value>)' },
        fg:      { DEFAULT: 'rgb(var(--c-fg) / <alpha-value>)', 0: 'rgb(var(--c-fg-0) / <alpha-value>)', 1: 'rgb(var(--c-fg-1) / <alpha-value>)', 2: 'rgb(var(--c-fg-2) / <alpha-value>)', 3: 'rgb(var(--c-fg-3) / <alpha-value>)' },
        line:    'rgb(var(--c-line) / <alpha-value>)',
        line2:   'rgb(var(--c-line2) / <alpha-value>)',
        accent:  { DEFAULT: 'rgb(var(--c-accent) / <alpha-value>)', hover: 'rgb(var(--c-accent-hover) / <alpha-value>)', dark: 'rgb(var(--c-accent-dark) / <alpha-value>)' },
        plantae: { green:  'rgb(var(--c-plantae-green) / <alpha-value>)', yellow: 'rgb(var(--c-plantae-yellow) / <alpha-value>)', red: 'rgb(var(--c-plantae-red) / <alpha-value>)' },
        score:   { green:  'rgb(var(--c-score-green) / <alpha-value>)', red: 'rgb(var(--c-score-red) / <alpha-value>)' },
        stage: {
          search:        'rgb(var(--c-stage-search) / <alpha-value>)',
          species:       'rgb(var(--c-stage-species) / <alpha-value>)',
          cluster:       'rgb(var(--c-stage-cluster) / <alpha-value>)',
          kingdom:       'rgb(var(--c-stage-kingdom) / <alpha-value>)',
          yolo:          'rgb(var(--c-stage-yolo) / <alpha-value>)',
          'bbox-bioclip': 'rgb(var(--c-stage-bbox) / <alpha-value>)',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'monospace'],
        sans: ['system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
    },
  },
  plugins: [],
};