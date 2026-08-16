/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      // Custom palette used across the dark UI. Keys are referenced as
      // e.g. bg-surface-2 or text-fg-2 in className strings.
      colors: {
        bg:      { DEFAULT: '#111', 0: '#0a0a0a', 1: '#14181d', 2: '#1a1a1a' },
        fg:      { DEFAULT: '#eee', 0: '#aaa', 1: '#888', 2: '#777', 3: '#444' },
        line:    '#222',
        line2:   '#2a2a2a',
        accent:  { DEFAULT: '#5cc078', hover: '#6fd089', dark: '#2c8a4a' },
        plantae: { green:  '#1a3a1a', yellow: '#3a3a1a', red:    '#3a1a1a' },
        score:   { green:  '#2c8a4a', red:    '#a35' },
        stage: {
          search:       '#4a9eff',
          species:      '#50c878',
          cluster:      '#f5a623',
          kingdom:      '#9b59b6',
          yolo:         '#e74c3c',
          'bbox-bioclip':'#1abc9c',
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