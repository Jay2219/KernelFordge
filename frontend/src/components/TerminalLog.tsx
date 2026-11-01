"use client";

import React, { useRef, useEffect } from "react";
import { Terminal, CheckCircle2, AlertTriangle, Activity } from "lucide-react";

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
        <span className="px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[10px] font-semibold">
          ARCHITECT: Muse Wonder
        </span>
      );
    }
    if (lower.includes("synthesizer") || (lower.includes("code") && !lower.includes("reflection"))) {
      return (
        <span className="px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 text-[10px] font-semibold">
          SYNTHESIZER: Muse Code
        </span>
      );
    }
    if (lower.includes("auditor") || lower.includes("baseten") || lower.includes("spark open")) {
      return (
        <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 text-[10px] font-semibold">
          AUDITOR: Muse Spark Open @ Baseten
        </span>
      );
    }
    if (lower.includes("reflection") || lower.includes("repair")) {
      return (
        <span className="px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 text-[10px] font-semibold">
          REFLECTOR: Muse Code (Self-Healing)
        </span>
      );
    }
    return (
      <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] font-semibold">
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
    <div className="h-full flex flex-col font-mono text-xs bg-[#06080D] overflow-hidden">
      {/* Drawer Header (Fixed 28px) */}
      <div className="h-7 border-b border-white/[0.06] bg-white/[0.02] px-3 flex items-center justify-between text-xs text-slate-400 shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-semibold tracking-wide text-slate-300">
            Closed-Loop Multi-Agent Compiler Trace
          </span>
        </div>
        <div className="flex items-center gap-2 font-mono text-[10px]">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-400">WebSocket Live [{steps.length} Events Recorded]</span>
        </div>
      </div>

      {/* Drawer Scrollable Content */}
      <div className="flex-1 p-2.5 overflow-y-auto space-y-1.5 bg-[#07090E]">
        {steps.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-600 text-[11px] gap-2">
            <span className="animate-pulse">&gt;</span> Standby. Select an operator preset and press &quot;Synthesize &amp; Compile&quot; (or Ctrl+Enter).
          </div>
        ) : (
          steps.map((step, idx) => {
            const timeTag = `[${new Date(Date.now() - (steps.length - idx) * 800).toTimeString().slice(0, 8)}]`;
            return (
              <div
                key={idx}
                className={`flex items-start gap-2 text-xs py-0.5 px-1.5 rounded transition-colors ${
                  step.status === "FAILED"
                    ? "bg-rose-950/20 text-rose-300"
                    : "text-slate-300 hover:bg-white/[0.02]"
                }`}
              >
                <span className="text-slate-500 font-mono text-[11px] shrink-0 mt-0.5">{timeTag}</span>
                <div className="shrink-0 mt-0.5">{getAgentBadge(step.agent_name)}</div>
                <div className="shrink-0 mt-0.5">{getStepIcon(step.status)}</div>
                <p className="flex-1 text-[11px] leading-snug break-words">
                  {step.message}
                </p>
              </div>
            );
          })
        )}
        <div ref={terminalEndRef} />
      </div>
    </div>
  );
};
