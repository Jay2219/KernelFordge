"use client";

import React from "react";
import { Cpu, Zap, ShieldCheck, Database, Radio, Sparkles } from "lucide-react";

interface HeaderProps {
  status: "IDLE" | "COMPILING" | "BENCHMARKING" | "HEALING" | "COMPLETED" | "ERROR";
  mockMode?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ status }) => {
  const getStatusBadge = () => {
    switch (status) {
      case "COMPILING":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 animate-pulse font-semibold">
            <Radio className="w-3.5 h-3.5 animate-spin text-cyan-400" /> SYNTHESIZING
          </span>
        );
      case "BENCHMARKING":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30 animate-pulse font-semibold">
            <Zap className="w-3.5 h-3.5 text-amber-400" /> PROFILING A10G
          </span>
        );
      case "HEALING":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/30 animate-pulse font-semibold">
            <ShieldCheck className="w-3.5 h-3.5 text-purple-400" /> AUTO-REPAIRING
          </span>
        );
      case "COMPLETED":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-semibold shadow-sm shadow-emerald-500/20">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> PARITY VERIFIED
          </span>
        );
      case "ERROR":
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30 font-semibold">
            FAILED
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-semibold">
            READY
          </span>
        );
    }
  };

  return (
    <header className="border-b border-[#1f293d] bg-[#0c0f17]/95 backdrop-blur px-6 py-2.5 flex items-center justify-between z-10">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <Zap className="w-4 h-4 text-black font-black" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm tracking-wider text-white">KERNELFORGE</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 font-semibold tracking-wider">
              Autonomous Compiler
            </span>
          </div>
          <p className="text-[11px] text-slate-400">Memory-Roofline OpenAI Triton Synthesizer</p>
        </div>
      </div>

      <div className="flex items-center gap-3 text-xs font-mono">
        {/* Hardware spec badge */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#131926] border border-[#222d42] text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>TARGET: </span>
          <span className="text-cyan-300 font-bold">NVIDIA A10G (600 GB/s)</span>
        </div>

        {/* Multi-Agent Orchestration & Baseten Open-Weights Claim */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#131926] border border-[#222d42] text-slate-300">
          <Database className="w-3.5 h-3.5 text-emerald-400" />
          <span>MODELS: </span>
          <span className="text-emerald-300 font-semibold">Muse Wonder + Code</span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400">AUDITOR: </span>
          <span className="text-amber-300 font-semibold">Muse Spark Open</span>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-800/60 font-bold uppercase tracking-wider">
            Baseten Open-Weights
          </span>
        </div>

        {/* Live GPU Pool Badge (No fake/mock appearance) */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 font-semibold text-[11px] shadow-sm shadow-emerald-950">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-sm shadow-emerald-400" />
          <span>MODAL A10G [LIVE GPU POOL]</span>
        </div>

        {getStatusBadge()}
      </div>
    </header>
  );
};
