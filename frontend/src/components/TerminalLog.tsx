"use client";

import React, { useRef, useEffect, useState } from "react";
import { Terminal, CheckCircle2, AlertTriangle, ShieldCheck, Zap, Activity } from "lucide-react";

export interface ExecutionStep {
  step_name: string;
  status: "PENDING" | "IN_PROGRESS" | "SUCCESS" | "FAILED";
  agent_name: string;
  message: string;
  code_snippet?: string;
  data?: any;
  timestamp?: string;
}

interface TerminalLogProps {
  steps: ExecutionStep[];
}

export const TerminalLog: React.FC<TerminalLogProps> = ({ steps }) => {
  const terminalEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps]);

  const getAgentBadge = (agent: string) => {
    const lower = agent.toLowerCase();
    if (lower.includes("wonder") || lower.includes("architect")) {
      return (
        <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-700/60 font-bold shadow-sm shadow-cyan-950">
          ARCHITECT: Muse Wonder
        </span>
      );
    }
    if (lower.includes("synthesizer") || (lower.includes("code") && !lower.includes("reflection"))) {
      return (
        <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700/60 font-bold shadow-sm shadow-emerald-950">
          SYNTHESIZER: Muse Code
        </span>
      );
    }
    if (lower.includes("auditor") || lower.includes("baseten") || lower.includes("spark open")) {
      return (
        <span className="text-[10px] px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-700/60 font-bold shadow-sm shadow-amber-950">
          AUDITOR: Muse Spark Open @ Baseten
        </span>
      );
    }
    if (lower.includes("reflection") || lower.includes("repair")) {
      return (
        <span className="text-[10px] px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-700/60 font-bold shadow-sm shadow-purple-950">
          REFLECTION: Muse Code (Self-Healing)
        </span>
      );
    }
    return (
      <span className="text-[10px] px-2 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-700/60 font-bold">
        SANDBOX: Modal A10G
      </span>
    );
  };

  const getStepIcon = (status: string) => {
    if (status === "SUCCESS") return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />;
    if (status === "FAILED") return <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />;
    return <Activity className="w-3.5 h-3.5 text-cyan-400 animate-spin shrink-0" />;
  };

  return (
    <div className="h-full flex flex-col font-mono text-xs border border-[#1f293d] rounded-lg bg-[#0a0d14] overflow-hidden">
      <div className="px-4 py-2 border-b border-[#1f293d] bg-[#0f1420] flex items-center justify-between text-slate-400">
        <span className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-semibold text-slate-300">CLOSED-LOOP MULTI-AGENT COMPILER TRACE</span>
        </span>
        <div className="flex items-center gap-3 text-[10px]">
          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            WEBSOCKET STREAM ACTIVE
          </span>
          <span className="text-slate-500 font-bold">[{steps.length} EVENTS]</span>
        </div>
      </div>

      <div className="flex-1 p-3 overflow-y-auto space-y-2">
        {steps.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-600 text-[11px] gap-2">
            <span className="animate-pulse">&gt;</span> Standby. Select an operator preset and click &quot;Synthesize &amp; Compile&quot; (or press Ctrl+Enter).
          </div>
        ) : (
          steps.map((step, idx) => (
            <div
              key={idx}
              className={`flex flex-col gap-1 border-l-2 pl-3 py-1 transition-colors ${
                step.status === "FAILED"
                  ? "border-rose-500/80 bg-rose-950/10"
                  : step.status === "SUCCESS"
                  ? "border-emerald-500/80 bg-emerald-950/5"
                  : "border-cyan-500/80 bg-cyan-950/5"
              }`}
            >
              <div className="flex items-center gap-2.5">
                {getStepIcon(step.status)}
                {getAgentBadge(step.agent_name)}
                <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                  [{step.step_name}]
                </span>
                <span className="text-[10px] text-slate-600 ml-auto">
                  T+{((idx + 1) * 0.4).toFixed(1)}s
                </span>
              </div>
              <p className={`text-[11px] leading-relaxed pl-6 ${
                step.status === "FAILED" ? "text-rose-300 font-mono" : "text-slate-300"
              }`}>
                {step.message}
              </p>
            </div>
          ))
        )}
        <div ref={terminalEndRef} />
      </div>
    </div>
  );
};
