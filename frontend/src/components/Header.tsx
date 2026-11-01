"use client";

import React from "react";
import { Cpu, Zap, ShieldCheck, Database, Radio } from "lucide-react";

interface HeaderProps {
  status: "IDLE" | "COMPILING" | "BENCHMARKING" | "HEALING" | "COMPLETED" | "ERROR";
  mockMode: boolean;
}

export const Header: React.FC<HeaderProps> = ({ status, mockMode }) => {
  const getStatusBadge = () => {
    switch (status) {
      case "COMPILING":
        return <span className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/30 animate-pulse"><Radio className="w-3.5 h-3.5 animate-spin" /> SYNTHESIZING</span>;
      case "BENCHMARKING":
        return <span className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full bg-yellow-500/10 text-yellow-400 border border-yellow-500/30 animate-pulse"><Zap className="w-3.5 h-3.5" /> PROFILING A10G</span>;
      case "HEALING":
        return <span className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/30 animate-pulse"><ShieldCheck className="w-3.5 h-3.5" /> SELF-HEALING</span>;
      case "COMPLETED":
        return <span className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"><ShieldCheck className="w-3.5 h-3.5" /> PARITY VERIFIED</span>;
      case "ERROR":
        return <span className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30">FAILED</span>;
      default:
        return <span className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full bg-slate-800 text-slate-400 border border-slate-700">STANDBY</span>;
    }
  };

  return (
    <header className="border-b border-[#1f293d] bg-[#0c0f17]/90 backdrop-blur px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <Zap className="w-5 h-5 text-black font-black" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-base tracking-wider text-white">KERNELFORGE</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 font-semibold tracking-wider">v0.1.0-alpha</span>
          </div>
          <p className="text-xs text-slate-400">Autonomous GPU Kernel Synthesizer & Roofline Compiler</p>
        </div>
      </div>

      <div className="flex items-center gap-3 text-xs font-mono">
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#131926] border border-[#222d42] text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>TARGET: </span>
          <span className="text-cyan-400 font-bold">NVIDIA A10G (600 GB/s)</span>
        </div>

        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#131926] border border-[#222d42] text-slate-300">
          <Database className="w-3.5 h-3.5 text-emerald-400" />
          <span>MODELS: </span>
          <span className="text-emerald-400 font-bold">Muse Wonder + Code + Spark</span>
        </div>

        {mockMode && (
          <div className="px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30 text-[11px] font-semibold">
            MOCK SANDBOX ACTIVE
          </div>
        )}

        {getStatusBadge()}
      </div>
    </header>
  );
};
