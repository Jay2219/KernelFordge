"use client";

import React, { useRef, useEffect } from "react";
import { Terminal, CheckCircle2, AlertTriangle, ShieldCheck, ArrowRight } from "lucide-react";

export interface ExecutionStep {
  step_name: string;
  status: "PENDING" | "IN_PROGRESS" | "SUCCESS" | "FAILED";
  agent_name: string;
  message: string;
  code_snippet?: string;
  data?: any;
}

interface TerminalLogProps {
  steps: ExecutionStep[];
}

export const TerminalLog: React.FC<TerminalLogProps> = ({ steps }) => {
  const terminalEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps]);

  const getAgentColor = (name: string) => {
    if (name.includes("Wonder")) return "text-cyan-400 bg-cyan-950/60 border-cyan-800/60";
    if (name.includes("Code") || name.includes("Synthesizer")) return "text-emerald-400 bg-emerald-950/60 border-emerald-800/60";
    if (name.includes("Auditor") || name.includes("Baseten")) return "text-amber-400 bg-amber-950/60 border-amber-800/60";
    if (name.includes("Reflection")) return "text-purple-400 bg-purple-950/60 border-purple-800/60";
    return "text-blue-400 bg-blue-950/60 border-blue-800/60";
  };

  const getStepIcon = (status: string) => {
    if (status === "SUCCESS") return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />;
    if (status === "FAILED") return <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />;
    return <div className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />;
  };

  return (
    <div className="h-full flex flex-col font-mono text-xs border border-[#1f293d] rounded-lg bg-[#0a0d14] overflow-hidden">
      <div className="px-4 py-2 border-b border-[#1f293d] bg-[#0f1420] flex items-center justify-between text-slate-400">
        <span className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span>CLOSED-LOOP MULTI-AGENT COMPILER TRACE</span>
        </span>
        <span className="text-[10px] text-slate-500">{steps.length} EVENTS RECORDED</span>
      </div>

      <div className="flex-1 p-3 overflow-y-auto space-y-2.5">
        {steps.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-600 text-[11px]">
            &gt; Standby. Select an operation preset and click &quot;Synthesize &amp; Compile&quot; to begin.
          </div>
        ) : (
          steps.map((step, idx) => (
            <div key={idx} className="flex flex-col gap-1 border-l-2 border-[#1e293b] pl-2.5 py-0.5">
              <div className="flex items-center gap-2">
                {getStepIcon(step.status)}
                <span className={`text-[10px] px-1.5 py-0.5 rounded border font-semibold ${getAgentColor(step.agent_name)}`}>
                  {step.agent_name}
                </span>
                <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                  [{step.step_name}]
                </span>
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed pl-5">
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

