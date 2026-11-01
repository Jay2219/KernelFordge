import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KernelForge // Autonomous GPU Kernel Compiler",
  description: "Closed-loop OpenAI Triton synthesizer, roofline memory compiler, and self-healing agent",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-[#090a0f] text-[#e2e8f0] min-h-screen">
        {children}
      </body>
    </html>
  );
}
