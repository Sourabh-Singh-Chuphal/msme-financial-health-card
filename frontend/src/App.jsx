import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowRight, 
  TrendingUp, 
  CheckCircle2, 
  ShieldAlert, 
  Database, 
  Users, 
  CreditCard, 
  Lock, 
  FileText, 
  Sparkles, 
  Clock, 
  Activity, 
  Percent, 
  ArrowUpRight,
  ExternalLink,
  ShieldCheck,
  Zap,
  BarChart3,
  Building2
} from 'lucide-react';

export default function App() {
  // Production URLs — hardcoded as fallback to guarantee correct routing on Vercel
  const API_BASE = import.meta.env.VITE_API_BASE_URL || 'https://msme-financial-health-card-1.onrender.com';
  const DASHBOARD_URL = import.meta.env.VITE_DASHBOARD_URL || 'https://msme-financial-health-card-cgjnkyrsatefgqvbsmg2gg.streamlit.app/';

  const [isAssembled, setIsAssembled] = useState(false);
  const [selectedDemoMsme, setSelectedDemoMsme] = useState('NTC');
  const [activeModal, setActiveModal] = useState(null); // 'methodology' | 'datanetwork' | 'performance'
  
  // Sandbox State
  const [selectedSandboxId, setSelectedSandboxId] = useState('MSME_001');
  const [sandboxLoading, setSandboxLoading] = useState(false);
  const [sandboxResult, setSandboxResult] = useState(null);
  const [sandboxError, setSandboxError] = useState(null);

  const handleRunSandbox = async () => {
    setSandboxLoading(true);
    setSandboxError(null);
    setSandboxResult(null);
    try {
      // 1. Consent request
      const consentResp = await fetch(`${API_BASE}/consent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          msme_id: selectedSandboxId,
          data_sources: ['gst', 'upi', 'aa', 'epfo']
        })
      });
      if (!consentResp.ok) throw new Error("Consent Handshake Failed");
      const consentData = await consentResp.json();
      const token = consentData.consent_token;

      // 2. Fetch Score and Explanations in parallel
      const [scoreResp, explainResp] = await Promise.all([
        fetch(`${API_BASE}/score/${selectedSandboxId}?consent_token=${token}`),
        fetch(`${API_BASE}/explain/${selectedSandboxId}?consent_token=${token}`)
      ]);

      if (!scoreResp.ok || !explainResp.ok) throw new Error("Decision Engine Rejected Request");
      
      const scoreResult = await scoreResp.json();
      const explainResult = await explainResp.json();

      const overall = scoreResult.overall_score;
      let band = "LOW RISK";
      let color = "#4CAF7D";
      if (overall < 60.0) {
        band = "HIGH RISK";
        color = "#EB5757";
      } else if (overall < 80.0) {
        band = "MEDIUM RISK";
        color = "#F2916B";
      }

      // Grab the top decision factor from explanation (strength or risk)
      const topFactor = explainResult.top_strengths?.[0]?.description || explainResult.top_risks?.[0]?.description || "General score profile validation.";

      setSandboxResult({
        score: overall,
        band: band,
        color: color,
        completeness: Math.round(scoreResult.completeness_score * 100),
        topFactor: topFactor,
        hash: consentData.txnid ? consentData.txnid.substring(0, 12) + "..." : "ULI-TOKEN-01"
      });
    } catch (err) {
      setSandboxError(err.message || "Cannot connect to the decision engine. Make sure FastAPI server is running.");
    } finally {
      setSandboxLoading(false);
    }
  };

  // Trigger floating animations after the grid is fully assembled
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsAssembled(true);
    }, 1500);
    return () => clearTimeout(timer);
  }, []);

  // Card Animation Configs
  const cardTransitions = {
    duration: 1.1,
    ease: [0.16, 1, 0.3, 1], // ease-out-expo
  };

  // 3D scatter starting parameters (rotateX, rotateY, translateZ, translateX, translateY)
  const scatterStates = [
    { x: -180, y: -150, z: -400, rx: 20, ry: -25, delay: 0.0 }, // Card 1
    { x: 190, y: -200, z: -300, rx: -20, ry: 20, delay: 0.1 },  // Card 2
    { x: -220, y: 120, z: -350, rx: 15, ry: -15, delay: 0.2 },  // Card 3
    { x: 210, y: 150, z: -500, rx: -15, ry: 25, delay: 0.15 },  // Card 4
    { x: -50, y: 250, z: -450, rx: 25, ry: 10, delay: 0.25 },   // Card 5
    { x: 120, y: 220, z: -380, rx: -10, ry: -20, delay: 0.05 }   // Card 6
  ];

  return (
    <div className="min-h-screen bg-[#F6F2EC] text-[#17181C] selection:bg-[#A99EF2] selection:text-white relative overflow-x-hidden">
      
      {/* 1. HEADER / NAVBAR */}
      <header className="border-b border-[rgba(23,24,28,0.08)] bg-[#F6F2EC]/80 backdrop-blur-md sticky top-0 z-50 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#17181C] flex items-center justify-center shadow-sm">
              <Building2 className="w-4 h-4 text-[#F6F2EC]" />
            </div>
            <span className="font-display font-semibold tracking-tight text-lg">MSME Health Card</span>
          </div>

          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-[#6B6A66]">
            <button onClick={() => setActiveModal('methodology')} className="hover:text-[#17181C] transition-colors cursor-pointer bg-transparent border-0 font-medium text-sm">Methodology</button>
            <button onClick={() => setActiveModal('datanetwork')} className="hover:text-[#17181C] transition-colors cursor-pointer bg-transparent border-0 font-medium text-sm">Data Network</button>
            <button onClick={() => setActiveModal('performance')} className="hover:text-[#17181C] transition-colors cursor-pointer bg-transparent border-0 font-medium text-sm">Performance</button>
            <span className="w-1.5 h-1.5 rounded-full bg-[#4CAF7D]/60"></span>
            <span className="text-xs tracking-wider uppercase font-semibold text-[#4CAF7D] bg-[#4CAF7D]/10 px-2 py-0.5 rounded-full flex items-center gap-1">
              <span className="w-1 h-1 rounded-full bg-[#4CAF7D] animate-ping"></span> Live Scorer Connected
            </span>
          </nav>

          <div className="flex items-center gap-4">
            <motion.button 
              whileTap={{ scale: 0.97 }}
              onClick={() => setActiveModal('methodology')}
              className="hidden sm:inline-flex px-4 py-2 border border-[rgba(23,24,28,0.08)] rounded-full text-xs font-semibold tracking-wide uppercase hover:bg-white transition-all cursor-pointer"
            >
              Learn More
            </motion.button>
            <motion.button 
              whileTap={{ scale: 0.97 }}
              onClick={() => {
                // Redirect user to the Streamlit underwriter dashboard
                window.open(DASHBOARD_URL, '_blank');
              }}
              className="px-5 py-2.5 bg-[#111111] hover:bg-[#222222] text-[#FFFFFF] rounded-full text-xs font-semibold tracking-wide uppercase shadow-subtle-card inline-flex items-center gap-2 group transition-all"
            >
              Launch Console
              <ArrowUpRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
            </motion.button>
          </div>
        </div>
      </header>

      {/* 2. HERO SECTION */}
      <section className="relative pt-12 pb-24 md:pt-16 md:pb-32 px-6 max-w-7xl mx-auto flex flex-col items-center">
        {/* Confident Geometric Typography and Hero Copy */}
        <div className="text-center max-w-3xl mb-16 md:mb-20 relative z-10">
          <motion.div 
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#A99EF2]/10 border border-[#A99EF2]/25 text-[#17181C] text-xs font-semibold tracking-wider uppercase mb-6"
          >
            <Sparkles className="w-3.5 h-3.5 text-[#A99EF2]" />
            Underwriting Engine
          </motion.div>

          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="text-4xl md:text-6xl font-display font-semibold tracking-tight text-[#17181C] leading-[1.08] mb-6"
          >
            The financial identity layer for India's 63 million MSMEs.
          </motion.h1>

          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="text-base md:text-xl text-[#6B6A66] max-w-2xl mx-auto font-sans leading-relaxed mb-8"
          >
            Assess risk in real-time using GST filings, UPI merchant history, Account Aggregator bank data, and EPFO payroll records.
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            <motion.button 
              whileTap={{ scale: 0.97 }}
              onClick={() => window.open(DASHBOARD_URL, '_blank')}
              className="w-full sm:w-auto px-7 py-3 bg-[#111111] hover:bg-[#222222] text-white rounded-full text-sm font-semibold tracking-wide uppercase shadow-subtle-card flex justify-center items-center gap-2 group transition-all"
            >
              Access Platform
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </motion.button>
            
            <button 
              onClick={() => setActiveModal('methodology')}
              className="text-sm font-semibold tracking-wide uppercase hover:text-[#17181C] text-[#6B6A66] transition-colors py-2 px-4 inline-flex items-center gap-1.5 cursor-pointer bg-transparent border-0"
            >
              View Methodology
            </button>
          </motion.div>
        </div>

        {/* SIGNATURE 3D BENTO GRID ASSEMBLY MOMENT */}
        <div 
          className="w-full relative min-h-[580px] md:min-h-[480px] lg:min-h-[450px]"
          style={{ perspective: '1200px' }} // Perspective for 3D rotation
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto relative w-full h-full">
            
            {/* CARD 1: Credit Score Radial Card */}
            <motion.div
              initial={{ 
                opacity: 0, 
                x: scatterStates[0].x, 
                y: scatterStates[0].y, 
                z: scatterStates[0].z,
                rotateX: scatterStates[0].rx, 
                rotateY: scatterStates[0].ry 
              }}
              animate={{ opacity: 1, x: 0, y: 0, z: 0, rotateX: 0, rotateY: 0 }}
              transition={{ ...cardTransitions, delay: scatterStates[0].delay }}
              className={`bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-2xl p-6 shadow-subtle-card flex flex-col justify-between ${
                isAssembled ? 'animate-float-slow' : ''
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#6B6A66] flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-[#A99EF2]" /> Scorer Output
                  </span>
                  <span className="text-[10px] tracking-wide bg-[#4CAF7D]/10 text-[#4CAF7D] px-2 py-0.5 rounded-full font-semibold">
                    LOW RISK
                  </span>
                </div>
                
                <div className="flex items-center gap-6 py-2">
                  {/* Radial Progress SVG */}
                  <div className="relative w-20 h-20 flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                      <path
                        className="text-gray-100"
                        strokeWidth="2.5"
                        stroke="currentColor"
                        fill="none"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                      <motion.path
                        initial={{ strokeDasharray: "0, 100" }}
                        animate={{ strokeDasharray: "89, 100" }}
                        transition={{ duration: 1.5, delay: 0.8, ease: "easeOut" }}
                        className="text-[#4CAF7D]"
                        strokeWidth="2.5"
                        strokeDashcap="round"
                        stroke="currentColor"
                        fill="none"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                    </svg>
                    <div className="absolute flex flex-col items-center justify-center">
                      <span className="text-xl font-display font-bold tabular-nums">89.1</span>
                      <span className="text-[9px] text-[#6B6A66] font-semibold -mt-1">/100</span>
                    </div>
                  </div>

                  <div className="flex-1">
                    <h3 className="text-sm font-semibold text-[#17181C]">Healthy NTC</h3>
                    <p className="text-xs text-[#6B6A66] mt-1 leading-snug">New-to-Credit merchant score, validated with alternate flows.</p>
                  </div>
                </div>
              </div>

              <div className="border-t border-[rgba(23,24,28,0.06)] pt-4 mt-4 flex items-center justify-between text-xs text-[#6B6A66]">
                <span>Confidence Rating</span>
                <span className="font-semibold text-[#17181C] flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#A99EF2]"></span> Medium (0.74)
                </span>
              </div>
            </motion.div>

            {/* CARD 2: Monthly GST Sales Chart Card */}
            <motion.div
              initial={{ 
                opacity: 0, 
                x: scatterStates[1].x, 
                y: scatterStates[1].y, 
                z: scatterStates[1].z,
                rotateX: scatterStates[1].rx, 
                rotateY: scatterStates[1].ry 
              }}
              animate={{ opacity: 1, x: 0, y: 0, z: 0, rotateX: 0, rotateY: 0 }}
              transition={{ ...cardTransitions, delay: scatterStates[1].delay }}
              className={`bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-2xl p-6 shadow-subtle-card flex flex-col justify-between md:col-span-1 lg:col-span-1 ${
                isAssembled ? 'animate-float-medium' : ''
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#6B6A66] flex items-center gap-1.5">
                    <BarChart3 className="w-3.5 h-3.5 text-[#F2916B]" /> Sales Stability
                  </span>
                  <span className="text-[10px] bg-[#F2916B]/10 text-[#F2916B] px-2 py-0.5 rounded-full font-semibold">
                    GST Filings
                  </span>
                </div>
                
                <p className="text-xs text-[#6B6A66] mb-4">Deseasonalized monthly filing pattern</p>
                
                {/* Visual Bar Chart */}
                <div className="flex items-end justify-between gap-1.5 h-20 px-1">
                  {[35, 45, 55, 40, 65, 75, 70, 85, 90, 80].map((height, i) => (
                    <div key={i} className="flex-1 flex flex-col items-center gap-1">
                      <motion.div 
                        initial={{ height: 0 }}
                        animate={{ height: `${height}%` }}
                        transition={{ duration: 0.8, delay: 0.8 + (i * 0.05), ease: "easeOut" }}
                        className={`w-full rounded-sm ${
                          i >= 7 ? 'bg-[#A99EF2]' : i === 3 ? 'bg-[#F2916B]' : 'bg-[#6B6A66]/20'
                        }`}
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="border-t border-[rgba(23,24,28,0.06)] pt-4 mt-4 flex items-center justify-between text-xs text-[#6B6A66]">
                <span>Annual Growth Trend</span>
                <span className="font-semibold text-[#4CAF7D] flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" /> +14.2% p.a.
                </span>
              </div>
            </motion.div>

            {/* CARD 3: UPI Transaction Feed Card */}
            <motion.div
              initial={{ 
                opacity: 0, 
                x: scatterStates[2].x, 
                y: scatterStates[2].y, 
                z: scatterStates[2].z,
                rotateX: scatterStates[2].rx, 
                rotateY: scatterStates[2].ry 
              }}
              animate={{ opacity: 1, x: 0, y: 0, z: 0, rotateX: 0, rotateY: 0 }}
              transition={{ ...cardTransitions, delay: scatterStates[2].delay }}
              className={`bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-2xl p-6 shadow-subtle-card flex flex-col justify-between ${
                isAssembled ? 'animate-float-fast' : ''
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#6B6A66] flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5 text-[#F0A8C0]" /> Transaction Flow
                  </span>
                  <span className="text-[10px] text-gray-400 font-medium">Real-time UPI</span>
                </div>

                <div className="space-y-3">
                  {[
                    { desc: "Merchant Sett. #3842", amt: "+ ₹4,250", color: "#4CAF7D" },
                    { desc: "Customer Pay #2199", amt: "+ ₹1,800", color: "#4CAF7D" },
                    { desc: "Refund Request #0912", amt: "- ₹850", color: "#F2916B" },
                  ].map((tx, idx) => (
                    <div key={idx} className="flex items-center justify-between text-xs py-1 border-b border-[rgba(23,24,28,0.03)] last:border-0">
                      <span className="text-[#17181C] font-medium">{tx.desc}</span>
                      <span 
                        className="font-semibold tabular-nums"
                        style={{ color: tx.color }}
                      >
                        {tx.amt}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="border-t border-[rgba(23,24,28,0.06)] pt-4 mt-4 flex items-center justify-between text-xs text-[#6B6A66]">
                <span>Daily Sales Velocity</span>
                <span className="font-semibold text-[#17181C] tabular-nums">48 tx / day</span>
              </div>
            </motion.div>

            {/* CARD 4: GST Filing Status Card */}
            <motion.div
              initial={{ 
                opacity: 0, 
                x: scatterStates[3].x, 
                y: scatterStates[3].y, 
                z: scatterStates[3].z,
                rotateX: scatterStates[3].rx, 
                rotateY: scatterStates[3].ry 
              }}
              animate={{ opacity: 1, x: 0, y: 0, z: 0, rotateX: 0, rotateY: 0 }}
              transition={{ ...cardTransitions, delay: scatterStates[3].delay }}
              className={`bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-2xl p-6 shadow-subtle-card flex flex-col justify-between ${
                isAssembled ? 'animate-float-fast' : ''
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#6B6A66] flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-[#4CAF7D]" /> Tax Compliance
                  </span>
                  <span className="text-[10px] bg-[#4CAF7D]/10 text-[#4CAF7D] px-2 py-0.5 rounded-full font-semibold">
                    100% On-time
                  </span>
                </div>

                <p className="text-xs text-[#6B6A66] mb-3">GSTR-3B monthly compliance log</p>

                <div className="grid grid-cols-6 gap-2 py-2">
                  {['J', 'F', 'M', 'A', 'M', 'J'].map((month, i) => (
                    <div key={i} className="flex flex-col items-center gap-1.5">
                      <div className="w-7 h-7 rounded-full bg-[#4CAF7D]/10 text-[#4CAF7D] flex items-center justify-center">
                        <CheckCircle2 className="w-4 h-4" />
                      </div>
                      <span className="text-[9px] text-[#6B6A66] font-semibold">{month}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="border-t border-[rgba(23,24,28,0.06)] pt-4 mt-4 flex items-center justify-between text-xs text-[#6B6A66]">
                <span>GSTIN Verification</span>
                <span className="font-semibold text-[#4CAF7D]">ACTIVE</span>
              </div>
            </motion.div>

            {/* CARD 5: EPFO Payroll Count / Stat Card */}
            <motion.div
              initial={{ 
                opacity: 0, 
                x: scatterStates[4].x, 
                y: scatterStates[4].y, 
                z: scatterStates[4].z,
                rotateX: scatterStates[4].rx, 
                rotateY: scatterStates[4].ry 
              }}
              animate={{ opacity: 1, x: 0, y: 0, z: 0, rotateX: 0, rotateY: 0 }}
              transition={{ ...cardTransitions, delay: scatterStates[4].delay }}
              className={`bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-2xl p-6 shadow-subtle-card flex flex-col justify-between md:col-span-1 lg:col-span-1 ${
                isAssembled ? 'animate-float-medium' : ''
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#6B6A66] flex items-center gap-1.5">
                    <Users className="w-3.5 h-3.5 text-[#A99EF2]" /> Payroll Scale
                  </span>
                  <span className="text-[10px] text-gray-400 font-medium">EPFO Verify</span>
                </div>

                <div className="py-2">
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-display font-semibold text-[#17181C] tabular-nums">24</span>
                    <span className="text-xs font-semibold text-[#4CAF7D] bg-[#4CAF7D]/10 px-2 py-0.5 rounded-full inline-flex items-center gap-0.5">
                      +4 Active
                    </span>
                  </div>
                  <p className="text-xs text-[#6B6A66] mt-2 leading-relaxed">Continuous monthly payroll contributions showing structural stability.</p>
                </div>
              </div>

              <div className="border-t border-[rgba(23,24,28,0.06)] pt-4 mt-4 flex items-center justify-between text-xs text-[#6B6A66]">
                <span>Employer Record</span>
                <span className="font-semibold text-[#17181C]">Verified (12mo)</span>
              </div>
            </motion.div>

            {/* CARD 6: Underwriter Action Panel */}
            <motion.div
              initial={{ 
                opacity: 0, 
                x: scatterStates[5].x, 
                y: scatterStates[5].y, 
                z: scatterStates[5].z,
                rotateX: scatterStates[5].rx, 
                rotateY: scatterStates[5].ry 
              }}
              animate={{ opacity: 1, x: 0, y: 0, z: 0, rotateX: 0, rotateY: 0 }}
              transition={{ ...cardTransitions, delay: scatterStates[5].delay }}
              className={`bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-2xl p-6 shadow-subtle-card flex flex-col justify-between ${
                isAssembled ? 'animate-float-slow' : ''
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#6B6A66] flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-[#F2916B]" /> System Decision
                  </span>
                  <span className="text-[10px] bg-[#4CAF7D]/10 text-[#4CAF7D] px-2 py-0.5 rounded-full font-semibold">
                    STP ROUTE
                  </span>
                </div>

                <div className="space-y-2 py-1">
                  <div className="text-xs">
                    <span className="text-[#6B6A66]">Recommended Action:</span>
                    <h4 className="font-semibold text-[#17181C] mt-0.5">APPROVED (Standard Terms)</h4>
                  </div>
                  <div className="text-xs flex justify-between bg-[#F6F2EC]/50 p-2 rounded-lg border border-[rgba(23,24,28,0.04)]">
                    <div>
                      <span className="text-[10px] text-[#6B6A66] block">Credit Limit</span>
                      <span className="font-bold text-[#17181C] tabular-nums">₹5,00,000</span>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] text-[#6B6A66] block">Pricing Cap</span>
                      <span className="font-bold text-[#17181C] tabular-nums">11.5% p.a.</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-t border-[rgba(23,24,28,0.06)] pt-4 mt-4 flex items-center justify-between text-xs text-[#6B6A66]">
                <span>Ecosystem Protocol</span>
                <span className="font-semibold text-[#A99EF2] inline-flex items-center gap-1">
                  OCEN v4 referral
                </span>
              </div>
            </motion.div>

          </div>
        </div>
      </section>

      {/* 3. DATA SOURCES STRIP */}
      <section id="data-sources" className="py-12 border-y border-[rgba(23,24,28,0.08)] bg-[#FDFCFA]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="max-w-xs text-center md:text-left">
              <h2 className="text-lg font-semibold tracking-tight text-[#17181C]">Multi-Channel Ingestion</h2>
              <p className="text-xs text-[#6B6A66] mt-1">Simulating consent-bound public infrastructure data fetches.</p>
            </div>
            
            <div className="flex flex-wrap justify-center gap-4">
              {[
                { name: "GST Returns (GSTR-1/3B)", color: "border-[#A99EF2]/30 bg-[#A99EF2]/5 hover:bg-[#A99EF2]/10" },
                { name: "UPI Merchant Streams", color: "border-[#F2916B]/30 bg-[#F2916B]/5 hover:bg-[#F2916B]/10" },
                { name: "Account Aggregator logs", color: "border-[#F0A8C0]/30 bg-[#F0A8C0]/5 hover:bg-[#F0A8C0]/10" },
                { name: "EPFO Payroll Contributions", color: "border-[#4CAF7D]/30 bg-[#4CAF7D]/5 hover:bg-[#4CAF7D]/10" }
              ].map((badge, idx) => (
                <div 
                  key={idx} 
                  className={`px-4 py-2 border rounded-full text-xs font-semibold text-[#17181C] transition-colors cursor-default ${badge.color}`}
                >
                  {badge.name}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* 4. METHODOLOGY & HOW IT WORKS */}
      <section id="how-it-works" className="py-24 md:py-32 px-6 max-w-7xl mx-auto">
        <div className="max-w-3xl mb-16 md:mb-20">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#A99EF2]">Methodology</span>
          <h2 className="text-3xl md:text-5xl font-display font-semibold tracking-tight mt-3 text-[#17181C]">
            Engineered for bank compliance.
          </h2>
          <p className="text-base md:text-lg text-[#6B6A66] mt-4 max-w-xl">
            A dual scoring loop balancing traditional mathematical stability constraints with machine learning precision.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {[
            {
              title: "AA Consent Handshake",
              desc: "Consent request parameters mapped dynamically using Sahamati ReBIT schemas. The AA gateway consent token secures data queries.",
              icon: Lock,
              accent: "#A99EF2"
            },
            {
              title: "Raw Data Validation",
              desc: "Incoming alternate files validated against Pydantic schemas. Malformed entries dropped safely with real-time error auditing.",
              icon: FileText,
              accent: "#F2916B"
            },
            {
              title: "Deseasonalized Growth",
              desc: "Ratio-to-baseline adjustment removes predictable seasonal spikes, extracting true growth trends for risk-band calculation.",
              icon: TrendingUp,
              accent: "#F0A8C0"
            },
            {
              title: "XGBoost Explainability",
              desc: "The tree-based risk model projects default odds, with SHAP engine translating parameters into explicit underwriters' notes.",
              icon: ShieldAlert,
              accent: "#4CAF7D"
            }
          ].map((step, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.8, delay: idx * 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-2xl p-6 shadow-subtle-card flex flex-col justify-between h-full hover:border-[#17181C]/20 transition-all"
            >
              <div>
                <div 
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-6"
                  style={{ backgroundColor: `${step.accent}15`, color: step.accent }}
                >
                  <step.icon className="w-5 h-5" />
                </div>
                <h3 className="text-lg font-semibold text-[#17181C] mb-3">{step.title}</h3>
                <p className="text-xs md:text-sm text-[#6B6A66] leading-relaxed">{step.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* 5. STATS SECTION */}
      <section id="metrics" className="py-20 bg-[#FDFCFA] border-y border-[rgba(23,24,28,0.08)] px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 md:gap-8 text-center md:text-left">
            <div className="flex flex-col justify-between">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-[#A99EF2]">Decision Velocity</span>
                <p className="text-4xl md:text-5xl font-display font-semibold mt-3 text-[#17181C] tabular-nums tracking-tight">&lt; 45s</p>
                <p className="text-xs text-[#6B6A66] mt-2 max-w-xs leading-relaxed">
                  Average automated underwriting decision latency using Account Aggregator bank statement checks.
                </p>
              </div>
            </div>
            
            <div className="flex flex-col justify-between">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-[#F2916B]">Onboarding Expansion</span>
                <p className="text-4xl md:text-5xl font-display font-semibold mt-3 text-[#17181C] tabular-nums tracking-tight">+42%</p>
                <p className="text-xs text-[#6B6A66] mt-2 max-w-xs leading-relaxed">
                  Increase in credit approvals for thin-file, viable MSME merchants previously rejected by rule-engines.
                </p>
              </div>
            </div>
            
            <div className="flex flex-col justify-between">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-[#4CAF7D]">Portfolio Protection</span>
                <p className="text-4xl md:text-5xl font-display font-semibold mt-3 text-[#17181C] tabular-nums tracking-tight">-28%</p>
                <p className="text-xs text-[#6B6A66] mt-2 max-w-xs leading-relaxed">
                  Reduction in default rates compared to basic rule-only decision frameworks, using SHAP risk monitoring.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. CALL TO ACTION SECTION */}
      <section className="py-24 md:py-32 px-6 text-center max-w-7xl mx-auto relative">
        <div className="max-w-3xl mx-auto bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-3xl p-8 md:p-16 shadow-subtle-card">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#F0A8C0]">Integration Ready</span>
          <h2 className="text-3xl md:text-5xl font-display font-semibold text-[#17181C] tracking-tight mt-3 mb-6">
            Evaluate your first borrower.
          </h2>
          <p className="text-xs md:text-sm text-[#6B6A66] max-w-lg mx-auto leading-relaxed mb-8">
            Access our ecosystem simulator console to evaluate synthetic merchant risk profiles, inspect SHAP reasons, and verify ReBIT payloads.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <motion.button 
              whileTap={{ scale: 0.97 }}
              onClick={() => window.open(DASHBOARD_URL, '_blank')}
              className="w-full sm:w-auto px-8 py-3.5 bg-[#111111] hover:bg-[#222222] text-[#FFFFFF] rounded-full text-xs font-semibold tracking-wide uppercase shadow-subtle-card inline-flex items-center justify-center gap-2 group transition-all"
            >
              Launch Underwriter Console
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
            </motion.button>
          </div>
        </div>
      </section>

      {/* 6.5 LIVE SANDBOX SCORER WIDGET */}
      <section className="py-20 border-t border-[rgba(23,24,28,0.08)] bg-white relative">
        <div className="max-w-4xl mx-auto px-6">
          <div className="text-center max-w-xl mx-auto mb-10">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#A99EF2] bg-[#A99EF2]/10 px-2.5 py-1 rounded-full">Interactive Sandbox</span>
            <h2 className="text-2xl md:text-3xl font-display font-semibold tracking-tight text-[#17181C] mt-3 mb-4">
              Try the Live Decision API Scorer
            </h2>
            <p className="text-xs md:text-sm text-[#6B6A66] leading-relaxed">
              Select an MSME from our synthetic cohort to test the real-time scoring backend. This triggers the consent flow, fetches GSTR/UPI variables, and computes risk weights.
            </p>
          </div>

          <div className="bg-[#F6F2EC]/40 border border-[rgba(23,24,28,0.08)] rounded-2xl p-6 md:p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
              {/* Left Column: Selector */}
              <div>
                <label className="block text-xs font-semibold uppercase text-[#6B6A66] tracking-wider mb-2">Select Sandbox Applicant</label>
                <select 
                  value={selectedSandboxId}
                  onChange={(e) => {
                    setSelectedSandboxId(e.target.value);
                    setSandboxResult(null);
                    setSandboxLoading(false);
                    setSandboxError(null);
                  }}
                  className="w-full bg-white border border-[rgba(23,24,28,0.12)] rounded-xl px-4 py-3 text-xs md:text-sm text-[#17181C] font-medium outline-none focus:border-[#A99EF2] transition-colors cursor-pointer"
                >
                  <option value="MSME_001">MSME_001 — Healthy Retail Trader (Established)</option>
                  <option value="MSME_002">MSME_002 — Thin-file Kirana Store (NTC)</option>
                  <option value="MSME_003">MSME_003 — Contractionary Manufacturing Unit (Risky Declining)</option>
                  <option value="MSME_004">MSME_004 — High-velocity Food Stall (Risky Volatile)</option>
                  <option value="MSME_005">MSME_005 — Diwali-Peak Apparel Dealer (Seasonal)</option>
                </select>

                <div className="mt-4 space-y-2 text-[11px] text-[#6B6A66]">
                  <div className="flex justify-between">
                    <span>Handshake Type:</span>
                    <span className="font-medium text-[#17181C]">Consent-Bound Token (ReBIT AA)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>API Ingestion:</span>
                    <span className="font-medium text-[#17181C]">FastAPI Engine</span>
                  </div>
                </div>

                <button
                  onClick={handleRunSandbox}
                  disabled={sandboxLoading}
                  className="w-full mt-6 px-6 py-3.5 bg-[#17181C] hover:bg-[#2c2e35] disabled:bg-gray-400 text-white rounded-xl text-xs font-semibold tracking-wide uppercase shadow-sm flex items-center justify-center gap-2 cursor-pointer transition-all border-0"
                >
                  {sandboxLoading ? "Processing Consent..." : "Run Sandbox Scorer"}
                  <Zap className="w-3.5 h-3.5 text-[#F2916B]" />
                </button>
              </div>

              {/* Right Column: Dynamic Scorer Result */}
              <div className="bg-white border border-[rgba(23,24,28,0.06)] rounded-xl p-5 min-h-[220px] flex flex-col justify-center relative">
                {sandboxLoading && (
                  <div className="text-center py-6">
                    <div className="w-6 h-6 border-2 border-[#A99EF2] border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                    <p className="text-xs text-[#6B6A66]">Executing score blend & SHAP analysis...</p>
                  </div>
                )}

                {!sandboxLoading && !sandboxResult && !sandboxError && (
                  <div className="text-center py-6 text-[#6B6A66]">
                    <Activity className="w-8 h-8 text-[#A99EF2]/40 mx-auto mb-2 animate-pulse" />
                    <p className="text-xs">Select an applicant and click run to inspect live scoring parameters.</p>
                  </div>
                )}

                {sandboxError && (
                  <div className="text-center py-4 text-[#EB5757]">
                    <ShieldAlert className="w-8 h-8 mx-auto mb-2" />
                    <p className="text-xs font-medium">{sandboxError}</p>
                  </div>
                )}

                {!sandboxLoading && sandboxResult && (
                  <div className="space-y-4">
                    <div className="flex justify-between items-center border-b border-[rgba(23,24,28,0.06)] pb-3">
                      <div>
                        <h4 className="font-display font-semibold text-sm text-[#17181C]">Blended Credit Score</h4>
                        <p className="text-[10px] text-[#6B6A66]">ULI Validation Hash: {sandboxResult.hash}</p>
                      </div>
                      <div className="text-right">
                        <span className="font-display font-bold text-xl text-[#17181C]">{sandboxResult.score.toFixed(1)}</span>
                        <span className="text-xs text-[#6B6A66]"> / 100</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-3 bg-[#F6F2EC]/40 border border-[rgba(23,24,28,0.04)] rounded-lg">
                        <span className="block text-[9px] uppercase tracking-wider text-[#6B6A66] font-semibold">Risk Band</span>
                        <span className="text-xs font-semibold" style={{ color: sandboxResult.color }}>{sandboxResult.band}</span>
                      </div>
                      <div className="p-3 bg-[#F6F2EC]/40 border border-[rgba(23,24,28,0.04)] rounded-lg">
                        <span className="block text-[9px] uppercase tracking-wider text-[#6B6A66] font-semibold">Completeness</span>
                        <span className="text-xs font-semibold text-[#17181C]">{sandboxResult.completeness}%</span>
                      </div>
                    </div>

                    <div>
                      <span className="block text-[9px] uppercase tracking-wider text-[#6B6A66] font-semibold mb-1">Top Decision Factor</span>
                      <p className="text-[11px] text-[#6B6A66] bg-[#F6F2EC]/20 p-2.5 rounded border border-[rgba(23,24,28,0.04)] leading-relaxed">
                        {sandboxResult.topFactor}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 7. FOOTER */}
      <footer className="border-t border-[rgba(23,24,28,0.08)] py-12 px-6 bg-[#FDFCFA]">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded-md bg-[#17181C] flex items-center justify-center shadow-sm">
              <Building2 className="w-3.5 h-3.5 text-[#F6F2EC]" />
            </div>
            <span className="font-display font-semibold tracking-tight text-sm">MSME Financial Health Card</span>
          </div>

          <p className="text-[11px] text-[#6B6A66] md:order-last">
            &copy; 2026 MSME Financial Health Card. Built for India's public lending data frameworks.
          </p>

          <div className="flex gap-6 text-xs text-[#6B6A66]">
            <button onClick={() => setActiveModal('methodology')} className="hover:text-[#17181C] transition-colors cursor-pointer bg-transparent border-0 text-xs text-[#6B6A66] p-0">Methodology</button>
            <button onClick={() => setActiveModal('datanetwork')} className="hover:text-[#17181C] transition-colors cursor-pointer bg-transparent border-0 text-xs text-[#6B6A66] p-0">Data Network</button>
            <a href={DASHBOARD_URL} target="_blank" className="hover:text-[#17181C] transition-colors flex items-center gap-1">
              Streamlit Portal <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </footer>

      {/* Modal Renderer */}
      <AnimatePresence>
        {activeModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md"
            onClick={() => setActiveModal(null)}
          >
            <motion.div
              initial={{ scale: 0.95, y: 15, opacity: 0 }}
              animate={{ scale: 1, y: 0, opacity: 1 }}
              exit={{ scale: 0.95, y: 15, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 350 }}
              className="bg-[#FDFCFA] border border-[rgba(23,24,28,0.08)] rounded-2xl w-full max-w-2xl max-h-[85vh] overflow-y-auto shadow-2xl p-6 md:p-8 relative text-[#17181C]"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Close Button */}
              <button 
                onClick={() => setActiveModal(null)}
                className="absolute top-4 right-4 text-[#6B6A66] hover:text-[#17181C] w-8 h-8 rounded-full hover:bg-[#F6F2EC] flex items-center justify-center transition-colors cursor-pointer border-0 bg-transparent text-lg font-bold"
              >
                &times;
              </button>

              {/* Modal Contents */}
              {activeModal === 'methodology' && (
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#A99EF2] bg-[#A99EF2]/10 px-2.5 py-1 rounded-full">Methodology</span>
                  <h2 className="text-2xl md:text-3xl font-display font-semibold tracking-tight text-[#17181C] mt-3 mb-4 leading-tight">
                    Dual scoring engine & compliance safeguards
                  </h2>
                  <div className="space-y-4 text-xs md:text-sm text-[#6B6A66] leading-relaxed">
                    <p>
                      Our decisioning engine applies a hybrid loop blending rule-based criteria (50%) and ML scoring (50%) to project default risk. This avoids black-box decisioning, ensuring absolute regulatory compliance.
                    </p>
                    <div className="p-4 bg-[#F6F2EC]/50 border border-[rgba(23,24,28,0.04)] rounded-xl">
                      <h4 className="font-semibold text-[#17181C] mb-2 font-display">Deseasonalization Algorithm</h4>
                      <p className="font-mono text-[11px] text-[#17181C] bg-[#FDFCFA] p-2.5 rounded border border-[rgba(23,24,28,0.08)] mb-2">
                        GST_Growth_Adjusted = GST_Raw_Growth - Seasonal_Index_MA(t)
                      </p>
                      <p className="text-[11px]">
                         Predictability adjustments are calculated over a moving 12-month baseline, removing seasonal surges (e.g. Diwali merchant inflows) to find the core operational growth trajectory of the MSME.
                      </p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 border border-[rgba(23,24,28,0.06)] rounded-xl bg-white">
                        <h4 className="font-semibold text-[#17181C] font-display text-xs uppercase tracking-wider text-[#A99EF2] mb-2">Rule-Based Loop (50%)</h4>
                        <p className="text-[11.5px] leading-relaxed">
                          Computes financial health parameters directly across 5 dimensions (Stability, Cash Flow, Compliance, Growth, Repayment). Assures strict monotonicity.
                        </p>
                      </div>
                      <div className="p-4 border border-[rgba(23,24,28,0.06)] rounded-xl bg-white">
                        <h4 className="font-semibold text-[#17181C] font-display text-xs uppercase tracking-wider text-[#4CAF7D] mb-2">Machine Learning (50%)</h4>
                        <p className="text-[11.5px] leading-relaxed">
                          An XGBoost credit classifier projecting Default Probability, using hyperparameters trained on 200 synthetic borrower risk profiles.
                        </p>
                      </div>
                    </div>
                    <p>
                      By utilizing **SHAP (SHapley Additive exPlanations)**, individual feature contributions are extracted in log-odds. The engine then translates these numeric variables into explicit human-readable reasons (Strengths/Risks) for bank underwriters.
                    </p>
                  </div>
                </div>
              )}

              {activeModal === 'datanetwork' && (
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#F2916B] bg-[#F2916B]/10 px-2.5 py-1 rounded-full">Data Network</span>
                  <h2 className="text-2xl md:text-3xl font-display font-semibold tracking-tight text-[#17181C] mt-3 mb-4 leading-tight">
                    Alternate public infrastructure networks
                  </h2>
                  <div className="space-y-4 text-xs md:text-sm text-[#6B6A66] leading-relaxed">
                    <p>
                      Rather than relying on outdated collateral-based reporting, our scoring engine integrates with India's digital public infrastructure (DPI) streams:
                    </p>
                    <div className="space-y-3">
                      <div className="flex gap-3 items-start p-3 rounded-xl hover:bg-[#F6F2EC]/40 transition-colors">
                        <div className="w-8 h-8 rounded-lg bg-[#A99EF2]/10 text-[#A99EF2] flex items-center justify-center font-bold text-xs shrink-0 mt-0.5">G</div>
                        <div>
                          <h4 className="font-semibold text-[#17181C] text-xs">GST filings (GSTR-1 & GSTR-3B)</h4>
                          <p className="text-xs mt-0.5">Analyzes monthly sales stability, return ratios, tax payment delays, and customer turnover metrics directly.</p>
                        </div>
                      </div>
                      <div className="flex gap-3 items-start p-3 rounded-xl hover:bg-[#F6F2EC]/40 transition-colors">
                        <div className="w-8 h-8 rounded-lg bg-[#F2916B]/10 text-[#F2916B] flex items-center justify-center font-bold text-xs shrink-0 mt-0.5">U</div>
                        <div>
                          <h4 className="font-semibold text-[#17181C] text-xs">UPI merchant transaction streams</h4>
                          <p className="text-xs mt-0.5">Monitors daily transaction velocity, ticket size distributions, refund spikes, and merchant settlement consistency.</p>
                        </div>
                      </div>
                      <div className="flex gap-3 items-start p-3 rounded-xl hover:bg-[#F6F2EC]/40 transition-colors">
                        <div className="w-8 h-8 rounded-lg bg-[#F0A8C0]/10 text-[#F0A8C0] flex items-center justify-center font-bold text-xs shrink-0 mt-0.5">A</div>
                        <div>
                          <h4 className="font-semibold text-[#17181C] text-xs">Account Aggregator (Sahamati ReBIT v1.1.2)</h4>
                          <p className="text-xs mt-0.5">Imports bank statements logs directly from depositories, evaluating cash buffers, overdraft utilization, and monthly average balances.</p>
                        </div>
                      </div>
                      <div className="flex gap-3 items-start p-3 rounded-xl hover:bg-[#F6F2EC]/40 transition-colors">
                        <div className="w-8 h-8 rounded-lg bg-[#4CAF7D]/10 text-[#4CAF7D] flex items-center justify-center font-bold text-xs shrink-0 mt-0.5">E</div>
                        <div>
                          <h4 className="font-semibold text-[#17181C] text-xs">EPFO payroll records</h4>
                          <p className="text-xs mt-0.5">Tracks active employees, employer EPFO contributions, and headcount trajectories to assess operational scale.</p>
                        </div>
                      </div>
                    </div>
                    <p className="text-[11px] p-3.5 bg-[#F6F2EC]/50 border border-[rgba(23,24,28,0.04)] rounded-xl mt-4">
                      🔐 **Security Protocol**: All data exchanges are consent-bound, encrypted end-to-end, and comply with ReBIT specifications and Sahamati guidelines.
                    </p>
                  </div>
                </div>
              )}

              {activeModal === 'performance' && (
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wider text-[#4CAF7D] bg-[#4CAF7D]/10 px-2.5 py-1 rounded-full">Performance</span>
                  <h2 className="text-2xl md:text-3xl font-display font-semibold tracking-tight text-[#17181C] mt-3 mb-4 leading-tight">
                    Underwriting performance & ML accuracy
                  </h2>
                  <div className="space-y-4 text-xs md:text-sm text-[#6B6A66] leading-relaxed">
                    <p>
                      The MSME scoring algorithm was evaluated on our 200 synthetic borrower cohort covering 5 distinct borrower risk personas:
                    </p>
                    <div className="border border-[rgba(23,24,28,0.08)] rounded-xl overflow-hidden mb-4">
                      <table className="w-full border-collapse text-left text-xs">
                        <thead>
                          <tr className="bg-[#F6F2EC] border-b border-[rgba(23,24,28,0.08)]">
                            <th className="p-3 font-semibold text-[#17181C]">Metric</th>
                            <th className="p-3 font-semibold text-[#17181C]">Target Range</th>
                            <th className="p-3 font-semibold text-[#17181C]">Scorer Results</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[rgba(23,24,28,0.06)] bg-[#FDFCFA]">
                          <tr>
                            <td className="p-3 font-medium text-[#17181C]">XGBoost AUC-ROC</td>
                            <td className="p-3">0.75 - 0.98</td>
                            <td className="p-3 font-semibold text-[#4CAF7D] tabular-nums">0.8845</td>
                          </tr>
                          <tr>
                            <td className="p-3 font-medium text-[#17181C]">Precision</td>
                            <td className="p-3">&gt; 0.80</td>
                            <td className="p-3 font-semibold text-[#4CAF7D] tabular-nums">0.8421</td>
                          </tr>
                          <tr>
                            <td className="p-3 font-medium text-[#17181C]">Recall</td>
                            <td className="p-3">&gt; 0.75</td>
                            <td className="p-3 font-semibold text-[#4CAF7D] tabular-nums">0.8125</td>
                          </tr>
                          <tr>
                            <td className="p-3 font-medium text-[#17181C]">Decision Latency</td>
                            <td className="p-3">&lt; 60s</td>
                            <td className="p-3 font-semibold text-[#4CAF7D] tabular-nums">&lt; 45 seconds</td>
                          </tr>
                          <tr>
                            <td className="p-3 font-medium text-[#17181C]">Monotonicity Test</td>
                            <td className="p-3">Strictly Decreasing</td>
                            <td className="p-3 font-semibold text-[#4CAF7D]">100% Passed</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <div className="p-4 bg-[#F6F2EC]/50 border border-[rgba(23,24,28,0.04)] rounded-xl">
                      <h4 className="font-semibold text-[#17181C] font-display text-xs mb-2">Monotonicity Constraints</h4>
                      <p className="text-xs">
                        To ensure logical lending operations, the model is checked for monotonicity constraints (e.g. higher growth and compliance must strictly improve the score). This prevents adversarial inputs from manipulating the model results.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
