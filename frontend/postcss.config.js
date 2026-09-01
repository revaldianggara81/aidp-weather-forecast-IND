// Polyfill for Turbopack's PostCSS sandbox, which doesn't expose structuredClone
if (typeof structuredClone === "undefined") {
  global.structuredClone = (obj) => JSON.parse(JSON.stringify(obj));
}

module.exports = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};
