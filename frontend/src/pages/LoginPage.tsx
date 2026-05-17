import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LoginPage() {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    navigate('/dashboard');
  };

  return (
    <div className="bg-surface min-h-screen flex items-center justify-center p-[20px] md:p-[40px] relative ai-pulse-bg text-on-surface">
      <main className="w-full max-w-[440px] relative z-10">
        <div className="glass-panel rounded-[24px] p-[32px] ambient-shadow border border-surface-container-highest/50 relative overflow-hidden">
          {/* Top accent line */}
          <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-secondary to-transparent opacity-30"></div>

          {/* Logo */}
          <div className="flex flex-col items-center mb-[32px]">
            <div className="w-[48px] h-[48px] bg-primary-container text-secondary rounded-full flex items-center justify-center mb-[8px]">
              <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1", fontSize: '28px' }}>account_balance</span>
            </div>
            <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface leading-[40px] tracking-tight">Agentic Bank</h1>
            <p className="font-[var(--font-body)] text-[16px] text-on-surface-variant mt-[4px] text-center leading-[24px]">Intelligent wealth management, secured.</p>
          </div>

          {/* Tab Toggle */}
          <div className="flex p-[4px] bg-surface-container-low rounded-[8px] mb-[16px]">
            <button
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-[8px] rounded-[6px] font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] text-center transition-all cursor-pointer border-none ${
                isLogin ? 'bg-surface-container-lowest shadow-sm text-on-surface' : 'bg-transparent text-on-surface-variant hover:text-on-surface'
              }`}
            >
              Log In
            </button>
            <button
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-[8px] rounded-[6px] font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] text-center transition-all cursor-pointer border-none ${
                !isLogin ? 'bg-surface-container-lowest shadow-sm text-on-surface' : 'bg-transparent text-on-surface-variant hover:text-on-surface'
              }`}
            >
              Sign Up
            </button>
          </div>

          {/* AI Login Banner */}
          <div className="bg-surface-bright border border-secondary/20 rounded-[12px] p-[8px] mb-[16px] relative overflow-hidden group hover:border-secondary/40 transition-colors">
            <div className="absolute inset-0 bg-secondary/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="flex items-center justify-between relative z-10">
              <div className="flex items-center gap-[12px]">
                <div className="w-[40px] h-[40px] rounded-full bg-primary-container flex items-center justify-center text-secondary">
                  <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>fingerprint</span>
                </div>
                <div>
                  <div className="flex items-center gap-[8px]">
                    <span className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em]">Secure AI Login</span>
                    <span className="px-[8px] py-[2px] bg-surface-container-high rounded-full font-[var(--font-mono)] text-[10px] text-secondary flex items-center gap-[4px]">
                      <span className="material-symbols-outlined" style={{ fontSize: '10px', fontVariationSettings: "'FILL' 1" }}>check_circle</span> Active
                    </span>
                  </div>
                  <span className="font-[var(--font-body)] text-[13px] text-on-surface-variant leading-tight block mt-[2px]">Behavioral &amp; biometric scanning</span>
                </div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input defaultChecked className="sr-only peer" type="checkbox" />
                <div className="w-[44px] h-[24px] bg-outline-variant peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-surface-container-lowest after:border-outline-variant after:border after:rounded-full after:h-[20px] after:w-[20px] after:transition-all peer-checked:bg-secondary"></div>
              </label>
            </div>
          </div>

          {/* Divider */}
          <div className="flex items-center gap-[16px] mb-[16px]">
            <div className="h-[1px] flex-1 bg-outline-variant/30"></div>
            <span className="font-[var(--font-mono)] text-[12px] text-outline uppercase tracking-[0.1em]">Or standard login</span>
            <div className="h-[1px] flex-1 bg-outline-variant/30"></div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="flex flex-col gap-[8px]">
            {!isLogin && (
              <div className="flex flex-col gap-[4px] animate-fade-in-up">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant px-[4px] tracking-[0.05em]">Full Name</label>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-[12px] top-1/2 -translate-y-1/2 text-outline-variant" style={{ fontSize: '20px' }}>person</span>
                  <input
                    className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] pl-[40px] pr-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant"
                    placeholder="Alex Rivers"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </div>
              </div>
            )}

            <div className="flex flex-col gap-[4px]">
              <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant px-[4px] tracking-[0.05em]">Email Address</label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-[12px] top-1/2 -translate-y-1/2 text-outline-variant" style={{ fontSize: '20px' }}>mail</span>
                <input
                  className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] pl-[40px] pr-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant"
                  placeholder="alex.rivers@example.com"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div className="flex flex-col gap-[4px]">
              <div className="flex justify-between items-center px-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Password</label>
                {isLogin && (
                  <button type="button" className="font-[var(--font-mono)] text-[12px] text-secondary hover:text-on-surface transition-colors bg-transparent border-none cursor-pointer tracking-[0.05em]">Forgot?</button>
                )}
              </div>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-[12px] top-1/2 -translate-y-1/2 text-outline-variant" style={{ fontSize: '20px' }}>lock</span>
                <input
                  className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] pl-[40px] pr-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant"
                  placeholder="••••••••••••"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full bg-secondary text-on-secondary font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] py-[14px] rounded-[8px] mt-[8px] hover:bg-on-surface-variant transition-colors flex items-center justify-center gap-[8px] cursor-pointer border-none"
            >
              {isLogin ? 'Access Vault' : 'Create Account'}
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>arrow_forward</span>
            </button>
          </form>

          {/* Footer */}
          <div className="mt-[16px] pt-[16px] border-t border-outline-variant/20 flex items-center justify-center gap-[8px] text-on-surface-variant">
            <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>security</span>
            <span className="font-[var(--font-mono)] text-[11px] uppercase tracking-[0.15em]">End-to-End Encrypted</span>
          </div>
        </div>
      </main>
    </div>
  );
}
