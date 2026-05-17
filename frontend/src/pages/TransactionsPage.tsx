import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const transactions = [
  { icon: 'storefront', name: 'Whole Foods', type: 'Groceries', amount: '-$142.50', status: 'AI Verified', statusIcon: 'check', date: 'Today, 2:14 PM' },
  { icon: 'person', name: 'Sarah Jenkins', type: 'Transfer', amount: '-$500.00', status: 'Pending Review', statusIcon: 'schedule', date: 'Today, 11:30 AM' },
  { icon: 'bolt', name: 'Electric Co.', type: 'Utilities', amount: '-$84.20', status: 'AI Verified', statusIcon: 'check', date: 'Yesterday, 9:00 PM' },
  { icon: 'local_cafe', name: 'Blue Bottle Coffee', type: 'Dining', amount: '-$12.80', status: 'AI Verified', statusIcon: 'check', date: 'Yesterday, 8:15 AM' },
  { icon: 'directions_car', name: 'Shell Gas', type: 'Transport', amount: '-$65.40', status: 'AI Verified', statusIcon: 'check', date: 'May 15, 5:22 PM' },
  { icon: 'shopping_bag', name: 'Amazon', type: 'Shopping', amount: '-$234.99', status: 'AI Verified', statusIcon: 'check', date: 'May 15, 1:00 PM' },
  { icon: 'work', name: 'Salary Deposit', type: 'Income', amount: '+$5,400.00', status: 'Completed', statusIcon: 'check', date: 'May 14, 12:00 AM' },
  { icon: 'fitness_center', name: 'Planet Fitness', type: 'Subscription', amount: '-$24.99', status: 'AI Verified', statusIcon: 'check', date: 'May 13, 6:00 AM' },
];

export default function TransactionsPage() {
  const navigate = useNavigate();
  const [filter, setFilter] = useState('all');

  return (
    <div className="p-[20px] md:p-[40px] max-w-[1000px] mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-[24px] gap-[16px]">
        <div>
          <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface tracking-tight">Transactions</h1>
          <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[4px]">AI-verified transaction history</p>
        </div>
        <div className="flex items-center gap-[8px]">
          <div className="flex items-center gap-[8px] bg-surface-container-high rounded-full px-[12px] py-[2px]">
            <span className="material-symbols-outlined text-secondary animate-pulse-glow" style={{ fontSize: '14px', fontVariationSettings: "'FILL' 1" }}>smart_toy</span>
            <span className="font-[var(--font-mono)] text-[11px] text-secondary tracking-[0.05em]">AI Fraud Agent Active</span>
          </div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-[8px] mb-[24px] overflow-x-auto pb-[4px]">
        {['all', 'verified', 'pending', 'income'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-[16px] py-[8px] rounded-full font-[var(--font-mono)] text-[12px] tracking-[0.05em] capitalize whitespace-nowrap transition-all cursor-pointer border-none ${
              filter === f ? 'bg-secondary text-on-secondary' : 'bg-surface-container-low text-on-surface-variant hover:bg-surface-container-high'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Send Funds CTA */}
      <div className="glass-panel rounded-[16px] p-[20px] ambient-shadow border border-secondary/10 mb-[24px] flex flex-col md:flex-row items-start md:items-center justify-between gap-[16px]">
        <div>
          <h3 className="font-[var(--font-headline)] text-[18px] font-semibold text-on-surface">Send Funds</h3>
          <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant mt-[2px]">Initiate secure transfers with real-time AI verification.</p>
        </div>
        <button
          onClick={() => navigate('/transfer')}
          className="bg-secondary text-on-secondary font-[var(--font-mono)] text-[12px] font-medium tracking-[0.05em] px-[20px] py-[10px] rounded-[8px] hover:bg-on-surface-variant transition-colors flex items-center gap-[8px] cursor-pointer border-none whitespace-nowrap"
        >
          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>swap_horiz</span>
          New Transfer
        </button>
      </div>

      {/* Transaction List */}
      <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 overflow-hidden">
        {transactions.map((tx, i) => (
          <div key={i} className="flex items-center justify-between py-[14px] px-[20px] border-b border-outline-variant/10 last:border-none hover:bg-surface-container-low/30 transition-colors cursor-pointer">
            <div className="flex items-center gap-[14px]">
              <div className="w-[40px] h-[40px] bg-primary-container rounded-[10px] flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined text-on-primary-container" style={{ fontSize: '20px' }}>{tx.icon}</span>
              </div>
              <div>
                <p className="font-[var(--font-body)] text-[14px] text-on-surface font-medium">{tx.name}</p>
                <p className="font-[var(--font-mono)] text-[11px] text-on-surface-variant tracking-[0.05em]">{tx.type} · {tx.date}</p>
              </div>
            </div>
            <div className="text-right shrink-0">
              <p className={`font-[var(--font-headline)] text-[14px] font-semibold ${tx.amount.startsWith('+') ? 'text-secondary' : 'text-on-surface'}`}>{tx.amount}</p>
              <div className="flex items-center gap-[4px] justify-end">
                <span className={`material-symbols-outlined ${tx.status === 'Pending Review' ? 'text-on-surface-variant' : 'text-secondary'}`} style={{ fontSize: '12px' }}>{tx.statusIcon}</span>
                <span className={`font-[var(--font-mono)] text-[10px] ${tx.status === 'Pending Review' ? 'text-on-surface-variant' : 'text-secondary'} tracking-[0.05em]`}>{tx.status}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
