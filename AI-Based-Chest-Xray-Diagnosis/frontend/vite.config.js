import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies API calls to the FastAPI server on :8000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") },
      "/files": "http://localhost:8000",
    },
  },
});
