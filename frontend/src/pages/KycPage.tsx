import { useState } from 'react';

export default function KycPage() {
  const [step, setStep] = useState(0);

  const verificationSteps = [
    { label: 'Personal Info', icon: 'person', done: true },
    { label: 'Document Upload', icon: 'upload_file', done: false },
    { label: 'Biometric Scan', icon: 'fingerprint', done: false },
    { label: 'AI Review', icon: 'smart_toy', done: false },
  ];

  return (
    <div className="p-[20px] md:p-[40px] max-w-[900px] mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-[32px] gap-[16px]">
        <div>
          <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface tracking-tight">KYC Verification</h1>
          <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px]">Complete your identity verification to unlock all features</p>
        </div>
        <div className="flex items-center gap-[8px] bg-surface-container-high rounded-full px-[12px] py-[6px]">
          <span className="material-symbols-outlined text-secondary animate-pulse-glow" style={{ fontSize: '14px', fontVariationSettings: "'FILL' 1" }}>verified_user</span>
          <span className="font-[var(--font-mono)] text-[11px] text-secondary tracking-[0.05em]">KYC Agent Active</span>
        </div>
      </div>

      {/* Progress Steps */}
      <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px] mb-[24px]">
        <div className="flex items-center justify-between mb-[4px]">
          {verificationSteps.map((s, i) => (
            <div key={i} className="flex flex-col items-center flex-1 relative">
              {i > 0 && (
                <div className={`absolute top-[18px] right-1/2 w-full h-[2px] -translate-x-0 ${i <= step ? 'bg-secondary' : 'bg-outline-variant/30'}`}></div>
              )}
              <div className={`w-[36px] h-[36px] rounded-full flex items-center justify-center relative z-10 transition-colors ${
                i <= step ? 'bg-secondary text-on-secondary' : 'bg-surface-container-high text-on-surface-variant'
              }`}>
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>{i < step ? 'check' : s.icon}</span>
              </div>
              <span className="font-[var(--font-mono)] text-[10px] text-on-surface-variant tracking-[0.05em] mt-[8px] text-center hidden md:block">{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      {step === 0 && (
        <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px] animate-fade-in-up">
          <h3 className="font-[var(--font-headline)] text-[18px] font-semibold text-on-surface mb-[20px]">Personal Information</h3>
          <form onSubmit={(e) => { e.preventDefault(); setStep(1); }} className="flex flex-col gap-[16px]">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-[16px]">
              <div className="flex flex-col gap-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">First Name</label>
                <input className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant" placeholder="Alex" defaultValue="Alex" />
              </div>
              <div className="flex flex-col gap-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Last Name</label>
                <input className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant" placeholder="Rivers" defaultValue="Rivers" />
              </div>
              <div className="flex flex-col gap-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Date of Birth</label>
                <input type="date" className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all" />
              </div>
              <div className="flex flex-col gap-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Nationality</label>
                <select className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all">
                  <option>United States</option>
                  <option>United Kingdom</option>
                  <option>Canada</option>
                  <option>Other</option>
                </select>
              </div>
            </div>
            <div className="flex flex-col gap-[4px]">
              <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Address</label>
              <input className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant" placeholder="123 Main St, Apt 4B, New York, NY 10001" />
            </div>
            <button type="submit" className="w-full bg-secondary text-on-secondary font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] py-[14px] rounded-[8px] hover:bg-on-surface-variant transition-colors flex items-center justify-center gap-[8px] cursor-pointer border-none">
              Continue
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>arrow_forward</span>
            </button>
          </form>
        </div>
      )}

      {step === 1 && (
        <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px] animate-fade-in-up">
          <h3 className="font-[var(--font-headline)] text-[18px] font-semibold text-on-surface mb-[20px]">Document Upload</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-[16px] mb-[20px]">
            {['Government ID (Front)', 'Government ID (Back)', 'Proof of Address', 'Selfie with ID'].map((doc) => (
              <label key={doc} className="flex flex-col items-center justify-center p-[24px] bg-surface-bright border-2 border-dashed border-outline-variant rounded-[12px] hover:border-secondary/40 transition-colors cursor-pointer group">
                <span className="material-symbols-outlined text-outline-variant group-hover:text-secondary mb-[8px] transition-colors" style={{ fontSize: '32px' }}>cloud_upload</span>
                <span className="font-[var(--font-body)] text-[13px] text-on-surface-variant text-center">{doc}</span>
                <span className="font-[var(--font-mono)] text-[10px] text-outline mt-[4px] tracking-[0.05em]">PNG, JPG up to 10MB</span>
                <input type="file" className="hidden" />
              </label>
            ))}
          </div>
          <div className="flex gap-[12px]">
            <button onClick={() => setStep(0)} className="flex-1 bg-surface-container-low text-on-surface font-[var(--font-mono)] text-[12px] tracking-[0.05em] py-[14px] rounded-[8px] cursor-pointer border-none">Back</button>
            <button onClick={() => setStep(2)} className="flex-1 bg-secondary text-on-secondary font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] py-[14px] rounded-[8px] cursor-pointer border-none flex items-center justify-center gap-[8px]">
              Continue
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>arrow_forward</span>
            </button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px] animate-fade-in-up">
          <h3 className="font-[var(--font-headline)] text-[18px] font-semibold text-on-surface mb-[20px]">Biometric Verification</h3>
          <div className="flex flex-col items-center py-[32px]">
            <div className="w-[120px] h-[120px] bg-primary-container rounded-full flex items-center justify-center mb-[20px] relative">
              <span className="material-symbols-outlined text-secondary" style={{ fontSize: '48px' }}>fingerprint</span>
              <div className="absolute inset-0 rounded-full border-2 border-secondary/30 animate-ping"></div>
            </div>
            <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant text-center mb-[24px]">Place your finger on the scanner or use facial recognition</p>
            <button onClick={() => setStep(3)} className="bg-secondary text-on-secondary font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] px-[32px] py-[14px] rounded-[8px] cursor-pointer border-none flex items-center gap-[8px]">
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>fingerprint</span>
              Start Scan
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[32px] flex flex-col items-center animate-fade-in-up">
          <div className="w-[64px] h-[64px] bg-secondary-container rounded-full flex items-center justify-center mb-[16px]">
            <span className="material-symbols-outlined text-secondary" style={{ fontSize: '32px', fontVariationSettings: "'FILL' 1" }}>check_circle</span>
          </div>
          <h3 className="font-[var(--font-headline)] text-[20px] font-semibold text-on-surface">Verification Complete!</h3>
          <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px] text-center max-w-[400px]">Your identity has been verified by our AI KYC Agent. All features are now unlocked.</p>
          <div className="mt-[16px] flex items-center gap-[8px] bg-secondary-container/30 px-[16px] py-[8px] rounded-full">
            <span className="material-symbols-outlined text-secondary" style={{ fontSize: '16px', fontVariationSettings: "'FILL' 1" }}>verified</span>
            <span className="font-[var(--font-mono)] text-[12px] text-secondary tracking-[0.05em]">KYC Level 3 — Full Access</span>
          </div>
        </div>
      )}
    </div>
  );
}
