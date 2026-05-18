import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getDashboard } from '../utils/api';
import type { DashboardData } from '../utils/api';

const quickActions = [
  { icon: 'swap_horiz', label: 'Transfer', path: '/transfer', color: 'text-secondary' },
  { icon: 'payments', label: 'Apply Loan', path: '/loans', color: 'text-secondary' },
  { icon: 'verified_user', label: 'Verify KYC', path: '/kyc', color: 'text-secondary' },
  { icon: 'security', label: 'Privacy', path: '/privacy', color: 'text-secondary' },
];

export default function DashboardPage() {
  const navigate = useNavigate();
  const [account, setAccount] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await getDashboard(1); // Hardcoded Alice Protocol
        setAccount(data);
      } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const transactions = account?.recent_transactions || [];

  return (
    <div className="p-[20px] md:p-[40px] max-w-[1200px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-[32px]">
        <div>
          <p className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em] uppercase mb-[4px]">Welcome back</p>
          <h1 className="font-[var(--font-headline)] text-[24px] md:text-[32px] font-semibold text-on-surface tracking-tight">
            {loading ? "Loading..." : account?.full_name || "Guest"}
          </h1>
        </div>
        <div className="flex items-center gap-[12px]">
          <button className="w-[40px] h-[40px] rounded-full bg-surface-container-high flex items-center justify-center text-on-surface-variant hover:text-on-surface transition-colors cursor-pointer border-none">
            <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>notifications</span>
          </button>
          <div className="w-[40px] h-[40px] rounded-full bg-primary-container border-2 border-secondary/30 overflow-hidden">
            <img
              alt="Profile"
              className="w-full h-full object-cover"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuDa3vfk84khTwpW8K7Mg7J5y_tcKxIJmTEPtfvNiLV1yljirvJtLBSNaNhKDMt4dJrzSPIv6_sfpO-wcRVx0SIRUbuqHnMjBXPy-eb2B7hf0PHPzJYPMbODQut5IqH5mFK5ZRtj96zK3fiY0v4bqQx1304AxGm1C2rQxSSisrqkfOXXes-81_dgXm6qGlBzNNddCyn0a13_9xALRgvvT-F66hFF0MnnaDKlN9Ad0BVtOF51dTpWtie8B5x1B0QU0jQ34Egk9l4y0MH9"
            />
          </div>
        </div>
      </div>

      {/* Balance Card */}
      <div className="glass-panel rounded-[20px] p-[24px] md:p-[32px] ambient-shadow border border-secondary/10 mb-[24px] relative overflow-hidden">
        <div className="absolute top-0 right-0 w-[200px] h-[200px] bg-secondary/5 rounded-full blur-[60px]"></div>
        <div className="relative z-10">
          <div className="flex items-center gap-[8px] mb-[8px]">
            <span className="material-symbols-outlined text-secondary" style={{ fontSize: '16px', fontVariationSettings: "'FILL' 1" }}>account_balance_wallet</span>
            <span className="font-[var(--font-mono)] text-[12px] text-on-surface-variant tracking-[0.05em] uppercase">Total Balance</span>
          </div>
          <h2 className="font-[var(--font-headline)] text-[36px] md:text-[48px] font-bold text-on-surface tracking-tight leading-tight">
            ${loading ? "..." : (account?.balance || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </h2>
          <div className="flex items-center gap-[4px] mt-[8px]">
            <span className="material-symbols-outlined text-secondary" style={{ fontSize: '16px' }}>trending_up</span>
            <span className="font-[var(--font-mono)] text-[12px] text-secondary tracking-[0.05em]">+2.4% this month</span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-[12px] mb-[32px]">
        {quickActions.map((action) => (
          <button
            key={action.path}
            onClick={() => navigate(action.path)}
            className="flex flex-col items-center gap-[8px] p-[16px] md:p-[20px] bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 hover:border-secondary/30 hover:shadow-lg transition-all cursor-pointer group"
          >
            <div className="w-[44px] h-[44px] bg-primary-container rounded-[12px] flex items-center justify-center group-hover:bg-secondary-container transition-colors">
              <span className={`material-symbols-outlined ${action.color}`} style={{ fontSize: '22px' }}>{action.icon}</span>
            </div>
            <span className="font-[var(--font-mono)] text-[12px] text-on-surface-variant font-medium tracking-[0.05em]">{action.label}</span>
          </button>
        ))}
      </div>

      {/* AI Status + Transactions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-[24px]">
        {/* AI Agent Status */}
        <div className="lg:col-span-1">
          <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[20px]">
            <div className="flex items-center gap-[8px] mb-[16px]">
              <span className="material-symbols-outlined text-secondary animate-pulse-glow" style={{ fontSize: '18px', fontVariationSettings: "'FILL' 1" }}>smart_toy</span>
              <span className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em] uppercase">AI Agents</span>
            </div>
            {[
              { name: 'Fraud Monitor', status: 'Active', icon: 'shield' },
              { name: 'Loan Processor', status: 'Idle', icon: 'payments' },
              { name: 'KYC Verifier', status: account?.kyc_status === 'verified' ? 'Completed' : account?.kyc_status === 'pending' ? 'Pending' : 'Active', icon: 'verified_user' },
              { name: 'Privacy Agent', status: 'Standby', icon: 'security' },
            ].map((agent) => (
              <div key={agent.name} className="flex items-center justify-between py-[10px] border-b border-outline-variant/10 last:border-none">
                <div className="flex items-center gap-[10px]">
                  <span className="material-symbols-outlined text-on-surface-variant" style={{ fontSize: '18px' }}>{agent.icon}</span>
                  <span className="font-[var(--font-body)] text-[14px] text-on-surface">{agent.name}</span>
                </div>
                <span className={`font-[var(--font-mono)] text-[10px] px-[8px] py-[3px] rounded-full tracking-[0.05em] ${
                  agent.status === 'Active' || agent.status === 'Completed' ? 'bg-secondary-container/40 text-secondary' :
                  agent.status === 'Idle' ? 'bg-surface-container-high text-on-surface-variant' :
                  agent.status === 'Pending' ? 'bg-error-container/40 text-error' :
                  'bg-tertiary-container/40 text-tertiary'
                }`}>
                  {agent.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Transactions */}
        <div className="lg:col-span-2">
          <div className="bg-surface-container-lowest rounded-[16px] border border-outline-variant/10 p-[20px]">
            <div className="flex items-center justify-between mb-[16px]">
              <div className="flex items-center gap-[8px]">
                <span className="material-symbols-outlined text-secondary" style={{ fontSize: '18px' }}>receipt_long</span>
                <span className="font-[var(--font-mono)] text-[12px] text-on-surface font-bold tracking-[0.05em] uppercase">Recent Transactions</span>
              </div>
              <button onClick={() => navigate('/transactions')} className="font-[var(--font-mono)] text-[12px] text-secondary tracking-[0.05em] bg-transparent border-none cursor-pointer hover:underline">View All</button>
            </div>
            {transactions.length === 0 && !loading ? (
              <p className="font-[var(--font-body)] text-[14px] text-on-surface-variant text-center py-[20px]">No recent transactions found.</p>
            ) : transactions.map((tx, i) => (
              <div key={i} className="flex items-center justify-between py-[12px] border-b border-outline-variant/10 last:border-none hover:bg-surface-container-low/30 rounded-[8px] px-[8px] transition-colors">
                <div className="flex items-center gap-[12px]">
                  <div className="w-[36px] h-[36px] bg-primary-container rounded-[10px] flex items-center justify-center">
                    <span className="material-symbols-outlined text-on-primary-container" style={{ fontSize: '18px' }}>{tx.icon}</span>
                  </div>
                  <div>
                    <p className="font-[var(--font-body)] text-[14px] text-on-surface font-medium">{tx.name}</p>
                    <p className="font-[var(--font-mono)] text-[11px] text-on-surface-variant tracking-[0.05em]">{tx.type}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-[var(--font-headline)] text-[14px] text-on-surface font-semibold">{tx.amount}</p>
                  <div className="flex items-center gap-[4px] justify-end">
                    <span className={`material-symbols-outlined ${tx.statusColor}`} style={{ fontSize: '12px' }}>{tx.statusIcon}</span>
                    <span className={`font-[var(--font-mono)] text-[10px] ${tx.statusColor} tracking-[0.05em]`}>{tx.status}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
