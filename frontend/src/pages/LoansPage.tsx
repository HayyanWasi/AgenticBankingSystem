import { useState, useEffect } from 'react';
import * as api from '../utils/api';

interface Loan {
  type: string;
  id: string;
  amount: string;
  remaining: string;
  status: string;
  progress: number;
}

export default function LoansPage() {
  const [step, setStep] = useState(0);
  const [loanType, setLoanType] = useState('');
  
  // Dynamic application inputs bound to Pydantic expectations
  const [amountInput, setAmountInput] = useState('');
  const [termMonths, setTermMonths] = useState(12);
  const [employmentStatus, setEmploymentStatus] = useState('Full-time');
  const [incomeInput, setIncomeInput] = useState('');
  const [purpose, setPurpose] = useState('');

  // Storage for active database rows
  const [activeLoans, setActiveLoans] = useState<Loan[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const loanProducts = [
    { type: 'Personal Loan', rate: '6.5%', max: '$50,000', icon: 'person', term: '1-5 years' },
    { type: 'Home Loan', rate: '4.2%', max: '$500,000', icon: 'home', term: '15-30 years' },
    { type: 'Auto Loan', rate: '5.1%', max: '$75,000', icon: 'directions_car', term: '2-7 years' },
    { type: 'Business Loan', rate: '7.8%', max: '$250,000', icon: 'business', term: '1-10 years' },
  ];

  // Fetch true credit rows on mount
  useEffect(() => {
    fetchLoans();
  }, []);

  const fetchLoans = async () => {
    try {
      const data = await api.get('/api/v1/loan/user/1'); // Fetch for User ID 1
      setActiveLoans(data);
    } catch (err) {
      console.error("Failed to sync credit profiles:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleApplicationSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError('');

    // Sanitize values into clean numerical floats and integers for the API
    const parsedAmount = parseFloat(amountInput.replace(/[^0-9.]/g, '')) || 0;
    const parsedIncome = parseFloat(incomeInput.replace(/[^0-9.]/g, '')) || 0;

    try {
      const res = await api.post('/api/v1/loan/apply', {
        full_name: "Alice Smith", // Shared context fallback
        id_card_num: "123456",    // Acts as Thread ID for LangGraph memory routing
        loan_amount: parsedAmount,
        loan_term_months: termMonths,
        monthly_income: parsedIncome / 12, // Converts annual to monthly income
        loan_purpose: loanType,
        loan_reason: purpose
      });

      // Shift to the completion checkmark UI page layout segment
      setStep(2);
      fetchLoans(); // Refresh historical lists in the background
    } catch (err: any) {
      setError(err.response?.data?.detail || "AI verification pipeline rejected submission parsing.");
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <div className="p-10 font-[var(--font-mono)] text-[14px]">Syncing Dynamic Credit States...</div>;

  return (
    <div className="p-[20px] md:p-[40px] max-w-[1000px] mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-[32px] gap-[16px]">
        <div>
          <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface tracking-tight">Loan Center</h1>
          <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px]">AI-powered loan processing &amp; management</p>
        </div>
        <div className="flex items-center gap-[8px] bg-surface-container-high rounded-full px-[12px] py-[6px]">
          <span className="material-symbols-outlined text-secondary animate-pulse-glow" style={{ fontSize: '14px', fontVariationSettings: "'FILL' 1" }}>smart_toy</span>
          <span className="font-[var(--font-mono)] text-[11px] text-secondary tracking-[0.05em]">Loan Agent Online</span>
        </div>
      </div>

      {error && <div className="p-4 mb-4 text-red-700 bg-red-100 rounded-lg text-[14px]">{error}</div>}

      {/* Dynamic Active Loans List Section */}
      {activeLoans.length > 0 && (
        <div className="mb-[32px]">
          <h2 className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em] uppercase mb-[16px]">Active Loans</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-[16px]">
            {activeLoans.map((loan, i) => (
              <div key={i} className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[20px] hover:border-secondary/20 transition-colors">
                <div className="flex items-center justify-between mb-[12px]">
                  <span className="font-[var(--font-body)] text-[15px] text-on-surface font-medium">{loan.type}</span>
                  <span className="font-[var(--font-mono)] text-[10px] px-[8px] py-[3px] rounded-full bg-secondary-container/40 text-secondary tracking-[0.05em]">{loan.status}</span>
                </div>
                <p className="font-[var(--font-mono)] text-[11px] text-on-surface-variant tracking-[0.05em] mb-[8px]">{loan.id}</p>
                <div className="flex justify-between mb-[8px]">
                  <span className="font-[var(--font-body)] text-[13px] text-on-surface-variant">Original: {loan.amount}</span>
                  <span className="font-[var(--font-headline)] text-[15px] text-on-surface font-semibold">Remaining: {loan.remaining}</span>
                </div>
                <div className="w-full h-[6px] bg-surface-container-high rounded-full overflow-hidden">
                  <div className="h-full bg-secondary rounded-full transition-all" style={{ width: `${loan.progress}%` }}></div>
                </div>
                <p className="font-[var(--font-mono)] text-[10px] text-on-surface-variant tracking-[0.05em] mt-[6px] text-right">{loan.progress}% paid off</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Apply for Loan Form Manager Switch Blocks */}
      <h2 className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em] uppercase mb-[16px]">Apply for a New Loan</h2>

      {step === 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-[12px]">
          {loanProducts.map((product) => (
            <button
              key={product.type}
              onClick={() => { setLoanType(product.type); setStep(1); }}
              className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[20px] hover:border-secondary/30 hover:shadow-lg transition-all cursor-pointer text-left group"
            >
              <div className="w-[44px] h-[44px] bg-primary-container rounded-[12px] flex items-center justify-center mb-[12px] group-hover:bg-secondary-container transition-colors">
                <span className="material-symbols-outlined text-secondary" style={{ fontSize: '22px' }}>{product.icon}</span>
              </div>
              <h3 className="font-[var(--font-headline)] text-[16px] font-semibold text-on-surface mb-[4px]">{product.type}</h3>
              <div className="flex flex-wrap gap-[8px] mt-[8px]">
                <span className="font-[var(--font-mono)] text-[10px] px-[8px] py-[3px] rounded-full bg-surface-container-high text-on-surface-variant tracking-[0.05em]">From {product.rate} APR</span>
                <span className="font-[var(--font-mono)] text-[10px] px-[8px] py-[3px] rounded-full bg-surface-container-high text-on-surface-variant tracking-[0.05em]">Up to {product.max}</span>
                <span className="font-[var(--font-mono)] text-[10px] px-[8px] py-[3px] rounded-full bg-surface-container-high text-on-surface-variant tracking-[0.05em]"> {product.term}</span>
              </div>
            </button>
          ))}
        </div>
      ) : step === 1 ? (
        <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px] animate-fade-in-up">
          <div className="flex items-center justify-between mb-[24px]">
            <h3 className="font-[var(--font-headline)] text-[18px] font-semibold text-on-surface">{loanType} Application</h3>
            <button type="button" onClick={() => setStep(0)} className="font-[var(--font-mono)] text-[12px] text-secondary bg-transparent border-none cursor-pointer tracking-[0.05em]">← Back</button>
          </div>

          <form onSubmit={handleApplicationSubmit} className="flex flex-col gap-[16px]">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-[16px]">
              <div className="flex flex-col gap-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Loan Amount ($)</label>
                <input 
                  className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] text-on-surface focus:outline-none" 
                  placeholder="25000" 
                  value={amountInput}
                  onChange={(e) => setAmountInput(e.target.value)}
                  required
                />
              </div>
              <div className="flex flex-col gap-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Loan Term</label>
                <select 
                  className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] text-on-surface focus:outline-none"
                  value={termMonths}
                  onChange={(e) => setTermMonths(parseInt(e.target.value))}
                >
                  <option value={12}>12 months</option>
                  <option value={24}>24 months</option>
                  <option value={36}>36 months</option>
                  <option value={48}>48 months</option>
                  <option value={60}>60 months</option>
                </select>
              </div>
              <div className="flex flex-col gap-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Employment Status</label>
                <select 
                  className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] text-on-surface focus:outline-none"
                  value={employmentStatus}
                  onChange={(e) => setEmploymentStatus(e.target.value)}
                >
                  <option>Full-time</option>
                  <option>Part-time</option>
                  <option>Self-employed</option>
                  <option>Retired</option>
                </select>
              </div>
              <div className="flex flex-col gap-[4px]">
                <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Annual Income ($)</label>
                <input 
                  className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] text-on-surface focus:outline-none" 
                  placeholder="85000" 
                  value={incomeInput}
                  onChange={(e) => setIncomeInput(e.target.value)}
                  required
                />
              </div>
            </div>
            <div className="flex flex-col gap-[4px]">
              <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em]">Purpose of Loan</label>
              <textarea 
                className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] text-on-surface focus:outline-none resize-none h-[80px]" 
                placeholder="Describe the purpose of this loan..." 
                value={purpose}
                onChange={(e) => setPurpose(e.target.value)}
                required
              />
            </div>
            <button 
              type="submit" 
              disabled={submitting}
              className="w-full bg-secondary text-on-secondary font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] py-[14px] rounded-[8px] hover:bg-on-surface-variant transition-colors flex items-center justify-center gap-[8px] cursor-pointer border-none disabled:opacity-50"
            >
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
                {submitting ? 'sync' : 'send'}
              </span>
              {submitting ? "Analyzing Credit with AI..." : "Submit Application"}
            </button>
          </form>
        </div>
      ) : (
        /* Step 2: Success State Output Screen Component */
        <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[32px] mt-[24px] flex flex-col items-center animate-fade-in-up">
          <div className="w-[64px] h-[64px] bg-secondary-container rounded-full flex items-center justify-center mb-[16px]">
            <span className="material-symbols-outlined text-secondary" style={{ fontSize: '32px', fontVariationSettings: "'FILL' 1" }}>check_circle</span>
          </div>
          <h3 className="font-[var(--font-headline)] text-[20px] font-semibold text-on-surface">Application Processed!</h3>
          <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px] text-center max-w-[400px]">
            Your evaluation data packet was run directly through the AI underwriting layer graph engine.
          </p>
          <button onClick={() => { setStep(0); }} className="mt-[20px] bg-primary-container text-on-primary-container font-[var(--font-mono)] text-[12px] px-[20px] py-[10px] rounded-[8px] cursor-pointer border-none tracking-[0.05em]">Back to Loan Center</button>
        </div>
      )}
    </div>
  );
}