import { useState, useEffect } from 'react';

export default function TransferPage() {
  const [recipient, setRecipient] = useState('');
  const [amount, setAmount] = useState('');
  const [note, setNote] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [resultMessage, setResultMessage] = useState('');
  const [resultStatus, setResultStatus] = useState<'success' | 'failed' | ''>('');
  const [senderAccount, setSenderAccount] = useState('');
  const [currentBalance, setCurrentBalance] = useState<number | null>(null);

  useEffect(() => {
    const fetchAccount = async () => {
      const userId = localStorage.getItem('user_id');
      if (userId) {
        try {
          const res = await fetch(`/api/v1/user/dashboard/${userId}`);
          if (res.ok) {
            const data = await res.json();
            if (data.account_number) {
              setSenderAccount(data.account_number);
              setCurrentBalance(data.balance);
            }
          }
        } catch (e) {
          console.error("Failed to fetch account info", e);
        }
      }
    };
    fetchAccount();
  }, []);

  const recentContacts = [
    { name: 'Sarah Jenkins', account: '****4521', avatar: 'person' },
    { name: 'Michael Chen', account: '****8834', avatar: 'person' },
    { name: 'Lisa Wang', account: '****2210', avatar: 'person' },
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!recipient || !amount) return;

    setLoading(true);
    setResultMessage('');
    setResultStatus('');

    if (!senderAccount) {
      setResultStatus('failed');
      setResultMessage('Could not determine your account number. Please log out and log in again.');
      setLoading(false);
      return;
    }

    try {
      const res = await fetch('/api/v1/transfer/initiate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sender_acc: senderAccount,
          receiver_acc: recipient,
          amount: parseFloat(amount),
        }),
      });

      const data = await res.json();

      if (!res.ok || data.status === 'failed') {
        setResultStatus('failed');
        setResultMessage(data.message || 'Transfer failed. Please try again.');
      } else {
        setResultStatus('success');
        setResultMessage(data.message || `Successfully sent $${amount} to ${recipient}.`);
        setSubmitted(true);
        setTimeout(() => {
          setSubmitted(false);
          setRecipient('');
          setAmount('');
          setNote('');
          setResultStatus('');
          setResultMessage('');
        }, 4000);
      }
    } catch {
      setResultStatus('failed');
      setResultMessage('Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-[20px] md:p-[40px] max-w-[800px] mx-auto">
      {/* Header */}
      <div className="mb-[32px]">
        <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface tracking-tight">Secure Transfer</h1>
        <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px]">
          Send money with AI-powered fraud protection
          {currentBalance !== null && (
             <span className="ml-[8px] font-bold text-secondary">| Available Balance: ${currentBalance.toFixed(2)}</span>
          )}
        </p>
      </div>

      {/* AI Status */}
      <div className="bg-surface-bright border border-secondary/20 rounded-[16px] p-[16px] mb-[24px] relative overflow-hidden">
        <div className="absolute inset-0 bg-secondary/5 opacity-40"></div>
        <div className="relative z-10 flex items-center gap-[12px]">
          <div className="w-[40px] h-[40px] rounded-full bg-primary-container flex items-center justify-center text-secondary">
            <span className="material-symbols-outlined animate-pulse-glow" style={{ fontSize: '20px', fontVariationSettings: "'FILL' 1" }}>shield</span>
          </div>
          <div>
            <span className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em]">AI Fraud Detection Active</span>
            <p className="font-[var(--font-body)] text-[13px] text-on-surface-variant mt-[2px]">Real-time pattern analysis on all outgoing transfers</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-[24px]">
        {/* Transfer Form */}
        <div className="lg:col-span-3">
          <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[24px]">
            <h3 className="font-[var(--font-headline)] text-[18px] font-semibold text-on-surface mb-[20px]">Transfer Details</h3>

            {submitted ? (
              <div className="flex flex-col items-center justify-center py-[40px] animate-fade-in-up">
                <div className="w-[64px] h-[64px] bg-secondary-container rounded-full flex items-center justify-center mb-[16px]">
                  <span className="material-symbols-outlined text-secondary" style={{ fontSize: '32px', fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                </div>
                <h3 className="font-[var(--font-headline)] text-[20px] font-semibold text-on-surface">Transfer Submitted</h3>
                <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px]">AI verification in progress...</p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="flex flex-col gap-[16px]">
                <div className="flex flex-col gap-[4px]">
                  <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant px-[4px] tracking-[0.05em]">Recipient</label>
                  <div className="relative">
                    <span className="material-symbols-outlined absolute left-[12px] top-1/2 -translate-y-1/2 text-outline-variant" style={{ fontSize: '20px' }}>person_search</span>
                    <input
                      className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] pl-[40px] pr-[16px] font-[var(--font-body)] text-[16px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant"
                      placeholder="Name or account number"
                      value={recipient}
                      onChange={(e) => setRecipient(e.target.value)}
                    />
                  </div>
                </div>

                <div className="flex flex-col gap-[4px]">
                  <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant px-[4px] tracking-[0.05em]">Amount (USD)</label>
                  <div className="relative">
                    <span className="material-symbols-outlined absolute left-[12px] top-1/2 -translate-y-1/2 text-outline-variant" style={{ fontSize: '20px' }}>attach_money</span>
                    <input
                      className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] pl-[40px] pr-[16px] font-[var(--font-headline)] text-[20px] font-semibold text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant placeholder:font-normal placeholder:text-[16px]"
                      placeholder="0.00"
                      type="number"
                      step="0.01"
                      value={amount}
                      onChange={(e) => setAmount(e.target.value)}
                    />
                  </div>
                </div>

                <div className="flex flex-col gap-[4px]">
                  <label className="font-[var(--font-mono)] text-[12px] text-on-surface-variant px-[4px] tracking-[0.05em]">Note (optional)</label>
                  <textarea
                    className="w-full bg-surface-bright border border-outline-variant rounded-[8px] py-[12px] px-[16px] font-[var(--font-body)] text-[14px] text-on-surface focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 transition-all placeholder:text-outline-variant resize-none h-[80px]"
                    placeholder="Add a note for the recipient..."
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                  />
                </div>

                {resultMessage && (
                  <div className={`rounded-[8px] p-[12px] text-[13px] font-[var(--font-body)] ${
                    resultStatus === 'success' 
                      ? 'bg-secondary-container/40 text-secondary border border-secondary/20' 
                      : 'bg-error-container/40 text-error border border-error/20'
                  }`}>
                    {resultMessage}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-secondary text-on-secondary font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] py-[14px] rounded-[8px] mt-[8px] hover:bg-on-surface-variant transition-colors flex items-center justify-center gap-[8px] cursor-pointer border-none disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <span className="material-symbols-outlined animate-spin" style={{ fontSize: '18px' }}>progress_activity</span>
                      Processing...
                    </>
                  ) : (
                    <>
                      <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>send</span>
                      Send Transfer
                    </>
                  )}
                </button>

              </form>
            )}
          </div>
        </div>

        {/* Recent Contacts */}
        <div className="lg:col-span-2">
          <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[20px]">
            <h3 className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em] uppercase mb-[16px]">Recent Contacts</h3>
            {recentContacts.map((contact, i) => (
              <button
                key={i}
                onClick={() => setRecipient(contact.name)}
                className="flex items-center gap-[12px] w-full py-[10px] px-[8px] rounded-[10px] hover:bg-surface-container-low transition-colors cursor-pointer bg-transparent border-none text-left"
              >
                <div className="w-[36px] h-[36px] bg-primary-container rounded-full flex items-center justify-center shrink-0">
                  <span className="material-symbols-outlined text-on-primary-container" style={{ fontSize: '18px' }}>{contact.avatar}</span>
                </div>
                <div>
                  <p className="font-[var(--font-body)] text-[14px] text-on-surface font-medium">{contact.name}</p>
                  <p className="font-[var(--font-mono)] text-[11px] text-on-surface-variant tracking-[0.05em]">{contact.account}</p>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
